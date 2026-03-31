try:
    from observability.observability_wrapper import (
        trace_agent, trace_step, trace_step_sync, trace_model_call, trace_tool_call,
    )
except ImportError:  # observability module not available (e.g. isolated test env)
    from contextlib import contextmanager as _obs_cm, asynccontextmanager as _obs_acm
    def trace_agent(*_a, **_kw):  # type: ignore[misc]
        def _deco(fn): return fn
        return _deco
    class _ObsHandle:
        output_summary = None
        def capture(self, *a, **kw): pass
    @_obs_acm
    async def trace_step(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    @_obs_cm
    def trace_step_sync(*_a, **_kw):  # type: ignore[misc]
        yield _ObsHandle()
    def trace_model_call(*_a, **_kw): pass  # type: ignore[misc]
    def trace_tool_call(*_a, **_kw): pass  # type: ignore[misc]

from modules.guardrails.content_safety_decorator import with_content_safety

GUARDRAILS_CONFIG = {'check_credentials_output': True,
 'check_jailbreak': True,
 'check_output': True,
 'check_pii_input': True,
 'check_toxic_code_output': True,
 'check_toxicity': True,
 'content_safety_enabled': True,
 'content_safety_severity_threshold': 2,
 'runtime_enabled': True,
 'sanitize_pii': False}


import os
import logging
import asyncio
import time as _time
from typing import Any, Dict, List, Optional, Union
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator, ValidationError, Field
from dotenv import load_dotenv
from email_validator import validate_email, EmailNotValidError
from PIL import Image
import numpy as np

# External dependencies (face_recognition, requests, etc.)
import requests

# Observability wrappers (trace_step, trace_step_sync, etc.) are injected by the runtime

# Load .env if present
load_dotenv()

# Logging configuration
logger = logging.getLogger("attendance_agent")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s"
)
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)

# --- Configuration Management ---

class Config:
    """Configuration loader for API keys and service endpoints."""
    @staticmethod
    def get_openai_api_key() -> Optional[str]:
        return os.getenv("OPENAI_API_KEY")

    @staticmethod
    def get_hris_api_url() -> Optional[str]:
        return os.getenv("HRIS_API_URL")

    @staticmethod
    def get_face_recognition_api_url() -> Optional[str]:
        return os.getenv("FACE_RECOGNITION_API_URL")

    @staticmethod
    def get_email_notification_api_url() -> Optional[str]:
        return os.getenv("EMAIL_NOTIFICATION_API_URL")

    @staticmethod
    def get_shift_scheduler_api_url() -> Optional[str]:
        return os.getenv("SHIFT_SCHEDULER_API_URL")

    @staticmethod
    def validate() -> bool:
        # Optional: Validate all required configs are present
        missing = []
        if not Config.get_openai_api_key():
            missing.append("OPENAI_API_KEY")
        if not Config.get_hris_api_url():
            missing.append("HRIS_API_URL")
        if not Config.get_face_recognition_api_url():
            missing.append("FACE_RECOGNITION_API_URL")
        if not Config.get_email_notification_api_url():
            missing.append("EMAIL_NOTIFICATION_API_URL")
        if not Config.get_shift_scheduler_api_url():
            missing.append("SHIFT_SCHEDULER_API_URL")
        if missing:
            logger.warning(f"Missing configuration keys: {missing}")
            return False
        return True

# --- LLM Integration (OpenAI) ---

@with_content_safety(config=GUARDRAILS_CONFIG)
def get_llm_client():
    import openai
    api_key = Config.get_openai_api_key()
    if not api_key:
        raise ValueError("OPENAI_API_KEY not configured")
    return openai.AsyncOpenAI(api_key=api_key)

# --- Input Validation Models ---

MAX_TEXT_LENGTH = 50000

class TextInputModel(BaseModel):
    input_data: str = Field(..., description="Text input from user")
    input_type: str = Field(..., description="Type of input: 'text'")
    user_context: dict = Field(..., description="User context information")

    @field_validator("input_data")
    @classmethod
    def validate_input_data(cls, v):
        if not isinstance(v, str):
            raise ValueError("Input data must be a string.")
        v = v.strip()
        if not v:
            raise ValueError("Input data cannot be empty.")
        if len(v) > MAX_TEXT_LENGTH:
            raise ValueError(f"Input data exceeds {MAX_TEXT_LENGTH} characters.")
        return v

    @field_validator("input_type")
    @classmethod
    def validate_input_type(cls, v):
        if v != "text":
            raise ValueError("Only 'text' input_type is supported in this endpoint.")
        return v

class ImageInputModel(BaseModel):
    input_data: str = Field(..., description="Base64-encoded image data")
    input_type: str = Field(..., description="Type of input: 'image'")
    user_context: dict = Field(..., description="User context information")

    @field_validator("input_type")
    @classmethod
    def validate_input_type(cls, v):
        if v != "image":
            raise ValueError("Only 'image' input_type is supported in this endpoint.")
        return v

class AttendanceValidationModel(BaseModel):
    employee_id: str
    check_in_time: str
    shift_start_time: str
    input_source: str

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("employee_id is required and must be a string.")
        return v.strip()

    @field_validator("check_in_time", "shift_start_time")
    @classmethod
    def validate_times(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Time fields must be non-empty strings in ISO format.")
        return v.strip()

    @field_validator("input_source")
    @classmethod
    def validate_input_source(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("input_source is required and must be a string.")
        return v.strip()

class ReportGenerationModel(BaseModel):
    employee_id: str
    date_range: dict
    report_type: str

    @field_validator("employee_id")
    @classmethod
    def validate_employee_id(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("employee_id is required and must be a string.")
        return v.strip()

    @field_validator("report_type")
    @classmethod
    def validate_report_type(cls, v):
        if v not in {"daily", "weekly", "monthly"}:
            raise ValueError("report_type must be one of: daily, weekly, monthly.")
        return v

class NotificationModel(BaseModel):
    recipient: str
    message: str
    notification_type: str

    @field_validator("recipient")
    @classmethod
    def validate_recipient(cls, v):
        try:
            validate_email(v)
        except EmailNotValidError as e:
            raise ValueError(f"Invalid email address: {str(e)}")
        return v

    @field_validator("message")
    @classmethod
    @with_content_safety(config=GUARDRAILS_CONFIG)
    def validate_message(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Message must be a non-empty string.")
        if len(v) > 10000:
            raise ValueError("Message too long.")
        return v.strip()

    @field_validator("notification_type")
    @classmethod
    def validate_notification_type(cls, v):
        if v not in {"alert", "confirmation"}:
            raise ValueError("notification_type must be 'alert' or 'confirmation'.")
        return v

class UserCredentialsModel(BaseModel):
    username: str
    password: str
    two_factor_code: Optional[str] = None

    @field_validator("username", "password")
    @classmethod
    def validate_non_empty(cls, v):
        if not v or not isinstance(v, str):
            raise ValueError("Field must be a non-empty string.")
        return v.strip()

# --- Supporting Classes ---

class InputProcessor:
    """Handles and parses incoming user inputs (text, images, badge scans)."""
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_text_input(self, text: str, user_context: dict) -> Dict[str, Any]:
        # LLM-based intent/entity extraction
        async with trace_step(
            "parse_text_input", step_type="parse",
            decision_summary="Extract intent and entities from user message",
            output_fn=lambda r: f"intent={r.get('intent','?')}, entities={r.get('entities','?')}"
        ) as step:
            intent, entities = await self._extract_intent_entities(text, user_context)
            result = {"intent": intent, "entities": entities}
            step.capture(result)
            return result

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_image_input(self, image_data: str, user_context: dict) -> Dict[str, Any]:
        # Call FaceRecognition API (mocked for now)
        async with trace_step(
            "process_image_input", step_type="tool_call",
            decision_summary="Verify identity via FaceRecognition",
            output_fn=lambda r: f"face_verified={r.get('face_verified', False)}"
        ) as step:
            result = await self._call_face_recognition(image_data, user_context)
            step.capture(result)
            return result

    async def _extract_intent_entities(self, text: str, user_context: dict) -> (str, dict):
        # Use LLM to extract intent/entities
        client = get_llm_client()
        system_prompt = (
            "You are an attendance assistant. Extract the user's intent (e.g., check-in, check-out, absence, report request) "
            "and any relevant entities (date, time, reason, employee_id) from the following message. "
            "Return a JSON object with 'intent' and 'entities'."
        )
        user_prompt = f"User message: {text}"
        try:
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=500
            )
            content = response.choices[0].message.content
            try:
                import json
                parsed = json.loads(content)
                intent = parsed.get("intent", "unknown")
                entities = parsed.get("entities", {})
            except Exception:
                intent = "unknown"
                entities = {}
            try:
                trace_model_call(
                    provider="openai", model_name="gpt-4o",
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    latency_ms=None,
                    response_summary=content[:200] if content else ""
                )
            except Exception:
                pass
            return intent, entities
        except Exception as e:
            logger.error(f"LLM intent extraction failed: {e}")
            return "unknown", {}

    async def _call_face_recognition(self, image_data: str, user_context: dict) -> Dict[str, Any]:
        # Simulate call to FaceRecognition API
        api_url = Config.get_face_recognition_api_url()
        if not api_url:
            logger.error("FaceRecognition API URL not configured.")
            return {"face_verified": False, "error": "FaceRecognition API not configured."}
        try:
            # In production, decode base64, send to API, etc.
            # Here, we simulate a successful verification
            # Simulate latency
            await asyncio.sleep(0.5)
            return {"face_verified": True, "employee_id": user_context.get("employee_id")}
        except Exception as e:
            logger.error(f"FaceRecognition API error: {e}")
            return {"face_verified": False, "error": str(e)}

class AttendanceValidator:
    """Validates attendance entries against company policies, shift schedules, and employee identity."""
    async def validate_check_in(self, employee_id: str, check_in_time: str, shift_start_time: str, input_source: str) -> Dict[str, Any]:
        async with trace_step(
            "validate_check_in", step_type="process",
            decision_summary="Validate check-in time and identity",
            output_fn=lambda r: f"status={r.get('status','?')}, errors={r.get('errors','')}"
        ) as step:
            # Business rule: Check-in within 15 minutes of shift start
            from datetime import datetime, timedelta
            errors = []
            try:
                check_in_dt = datetime.fromisoformat(check_in_time)
                shift_start_dt = datetime.fromisoformat(shift_start_time)
                delta = (check_in_dt - shift_start_dt).total_seconds() / 60
                if abs(delta) > 15:
                    errors.append("Check-in time is not within 15 minutes of shift start.")
                    status = "Late" if delta > 0 else "Early"
                else:
                    status = "On Time"
            except Exception as e:
                errors.append(f"Invalid time format: {e}")
                status = "Invalid"
            # Simulate identity validation (should call FaceRecognition or badge scan)
            if input_source not in {"face", "badge", "text"}:
                errors.append("Invalid input source.")
            result = {
                "status": status,
                "errors": errors,
                "employee_id": employee_id,
                "check_in_time": check_in_time,
                "shift_start_time": shift_start_time
            }
            step.capture(result)
            return result

    async def validate_identity(self, employee_id: str, user_context: dict) -> Dict[str, Any]:
        # Simulate identity validation
        async with trace_step(
            "validate_identity", step_type="process",
            decision_summary="Validate employee identity",
            output_fn=lambda r: f"identity_valid={r.get('identity_valid', False)}"
        ) as step:
            # In production, call FaceRecognition or badge scan
            identity_valid = user_context.get("employee_id") == employee_id
            result = {"identity_valid": identity_valid}
            step.capture(result)
            return result

class ReportGenerator:
    """Generates attendance reports (daily, weekly, monthly) for HR and management."""
    async def generate_report(self, employee_id: str, date_range: dict, report_type: str) -> Dict[str, Any]:
        async with trace_step(
            "generate_report", step_type="tool_call",
            decision_summary="Generate attendance report",
            output_fn=lambda r: f"report_type={report_type}, records={len(r.get('records',[]))}"
        ) as step:
            # Simulate HRIS API call
            api_url = Config.get_hris_api_url()
            if not api_url:
                logger.error("HRIS API URL not configured.")
                return {"success": False, "error": "HRIS API not configured."}
            try:
                # Simulate fetching records
                await asyncio.sleep(0.5)
                records = [
                    {"date": "2024-06-01", "status": "On Time"},
                    {"date": "2024-06-02", "status": "Late"},
                    {"date": "2024-06-03", "status": "Absent"},
                ]
                report = {
                    "employee_id": employee_id,
                    "report_type": report_type,
                    "date_range": date_range,
                    "records": records
                }
                step.capture(report)
                return {"success": True, "report": report}
            except Exception as e:
                logger.error(f"HRIS API error: {e}")
                return {"success": False, "error": str(e)}

class AnomalyDetector:
    """Detects and flags suspicious or inconsistent attendance patterns."""
    async def detect_anomalies(self, attendance_records: List[dict]) -> List[dict]:
        async with trace_step(
            "detect_anomalies", step_type="process",
            decision_summary="Detect suspicious attendance patterns",
            output_fn=lambda r: f"anomalies={len(r)}"
        ) as step:
            anomalies = []
            late_count = 0
            for rec in attendance_records:
                if rec.get("status") == "Late":
                    late_count += 1
                    if late_count > 2:
                        anomalies.append({
                            "type": "Repeated Late",
                            "details": rec
                        })
                if rec.get("status") == "Absent":
                    anomalies.append({
                        "type": "Absence",
                        "details": rec
                    })
            step.capture(anomalies)
            return anomalies

class NotificationManager:
    """Sends alerts and notifications to employees and HR via email or dashboard."""
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def send_alert(self, recipient: str, message: str) -> Dict[str, Any]:
        return await self._send_notification(recipient, message, "alert")

    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def send_confirmation(self, recipient: str, message: str) -> Dict[str, Any]:
        return await self._send_notification(recipient, message, "confirmation")

    async def _send_notification(self, recipient: str, message: str, notification_type: str) -> Dict[str, Any]:
        api_url = Config.get_email_notification_api_url()
        if not api_url:
            logger.error("EmailNotification API URL not configured.")
            return {"success": False, "error": "EmailNotification API not configured."}
        attempt = 0
        max_attempts = 3
        while attempt < max_attempts:
            try:
                # Simulate sending email
                await asyncio.sleep(0.2)
                # In production, send HTTP POST to email API
                return {"success": True, "delivery_status": "sent"}
            except Exception as e:
                logger.error(f"Notification send failed: {e}")
                attempt += 1
                await asyncio.sleep(2 ** attempt)
        logger.error("Notification send failed after 3 attempts.")
        return {"success": False, "error": "Failed to send notification after 3 attempts."}

class SecurityComplianceManager:
    """Handles authentication, authorization, PII masking, audit logging, and ensures compliance."""
    async def authenticate_user(self, user_credentials: dict) -> Dict[str, Any]:
        async with trace_step(
            "authenticate_user", step_type="process",
            decision_summary="Authenticate user via SSO and 2FA",
            output_fn=lambda r: f"authenticated={r.get('authenticated', False)}"
        ) as step:
            # Simulate SSO/2FA authentication
            username = user_credentials.get("username")
            password = user_credentials.get("password")
            two_factor_code = user_credentials.get("two_factor_code")
            # In production, call SSO/2FA provider
            if username == "testuser" and password == "testpass" and (not two_factor_code or two_factor_code == "123456"):
                user_context = {"employee_id": "E123", "username": username, "authenticated": True}
            else:
                user_context = {"authenticated": False}
            step.capture(user_context)
            return user_context

    async def authorize_action(self, user_context: dict, action: str) -> bool:
        # Simulate authorization logic
        return user_context.get("authenticated", False)

    async def log_event(self, event: dict):
        # Log event for audit/compliance
        logger.info(f"Audit log: {event}")

    @with_content_safety(config=GUARDRAILS_CONFIG)
    def mask_pii(self, data: dict) -> dict:
        # Mask PII fields for compliance
        masked = data.copy()
        if "employee_id" in masked:
            masked["employee_id"] = masked["employee_id"][:2] + "***"
        if "username" in masked:
            masked["username"] = masked["username"][0] + "***"
        return masked

# --- Main Agent Class ---

class EcommerceAttendanceTrackerAgent:
    """Main agent class for attendance tracking."""
    def __init__(self):
        self.input_processor = InputProcessor()
        self.attendance_validator = AttendanceValidator()
        self.report_generator = ReportGenerator()
        self.anomaly_detector = AnomalyDetector()
        self.notification_manager = NotificationManager()
        self.security_manager = SecurityComplianceManager()
        self.llm_model = "gpt-4o"
        self.llm_temperature = 0.7
        self.llm_max_tokens = 2000
        self.system_prompt = (
            "You are the Ecommerce Attendance Tracker Agent. Your role is to help employees and HR staff track, validate, "
            "and report attendance in a friendly, accurate, and policy-compliant manner. Always verify identity, follow business rules, "
            "and escalate issues as needed."
        )
        self.user_prompt_template = "Hi {user_name}, how can I assist you with your attendance today?"

    @trace_agent(agent_name='Ecommerce Attendance Tracker Agent edited')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def process_input(self, input_data: str, input_type: str, user_context: dict) -> Dict[str, Any]:
        async with trace_step(
            "process_input", step_type="parse",
            decision_summary="Route and process incoming user input",
            output_fn=lambda r: f"intent={r.get('intent','?')}"
        ) as step:
            if input_type == "text":
                parsed = await self.input_processor.process_text_input(input_data, user_context)
            elif input_type == "image":
                parsed = await self.input_processor.process_image_input(input_data, user_context)
            else:
                logger.error(f"Unsupported input_type: {input_type}")
                raise ValueError("Unsupported input_type. Supported: 'text', 'image'.")
            step.capture(parsed)
            return parsed

    async def validate_attendance_entry(self, employee_id: str, check_in_time: str, shift_start_time: str, input_source: str) -> Dict[str, Any]:
        async with trace_step(
            "validate_attendance_entry", step_type="process",
            decision_summary="Validate attendance entry against policy and schedule",
            output_fn=lambda r: f"status={r.get('status','?')}, errors={r.get('errors','')}"
        ) as step:
            result = await self.attendance_validator.validate_check_in(employee_id, check_in_time, shift_start_time, input_source)
            step.capture(result)
            return result

    async def generate_attendance_report(self, employee_id: str, date_range: dict, report_type: str) -> Dict[str, Any]:
        async with trace_step(
            "generate_attendance_report", step_type="tool_call",
            decision_summary="Generate attendance report for HR/management",
            output_fn=lambda r: f"success={r.get('success', False)}"
        ) as step:
            result = await self.report_generator.generate_report(employee_id, date_range, report_type)
            step.capture(result)
            return result

    async def detect_anomalies(self, attendance_records: List[dict]) -> List[dict]:
        async with trace_step(
            "detect_anomalies", step_type="process",
            decision_summary="Detect suspicious attendance patterns",
            output_fn=lambda r: f"anomalies={len(r)}"
        ) as step:
            anomalies = await self.anomaly_detector.detect_anomalies(attendance_records)
            step.capture(anomalies)
            return anomalies

    @trace_agent(agent_name='Ecommerce Attendance Tracker Agent edited')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def send_notification(self, recipient: str, message: str, notification_type: str) -> Dict[str, Any]:
        async with trace_step(
            "send_notification", step_type="tool_call",
            decision_summary="Send notification to user or HR",
            output_fn=lambda r: f"success={r.get('success', False)}"
        ) as step:
            if notification_type == "alert":
                result = await self.notification_manager.send_alert(recipient, message)
            elif notification_type == "confirmation":
                result = await self.notification_manager.send_confirmation(recipient, message)
            else:
                logger.error(f"Unsupported notification_type: {notification_type}")
                result = {"success": False, "error": "Unsupported notification_type."}
            step.capture(result)
            return result

    async def authenticate_user(self, user_credentials: dict) -> Dict[str, Any]:
        async with trace_step(
            "authenticate_user", step_type="process",
            decision_summary="Authenticate user via SSO and 2FA",
            output_fn=lambda r: f"authenticated={r.get('authenticated', False)}"
        ) as step:
            result = await self.security_manager.authenticate_user(user_credentials)
            step.capture(result)
            return result

    @trace_agent(agent_name='Ecommerce Attendance Tracker Agent edited')
    @with_content_safety(config=GUARDRAILS_CONFIG)
    async def chat_response(self, user_message: str, user_context: dict) -> Dict[str, Any]:
        async with trace_step(
            "chat_response", step_type="llm_call",
            decision_summary="Call LLM to generate user-facing response",
            output_fn=lambda r: f"length={len(r.get('response','')) if r.get('response') else 0}"
        ) as step:
            client = get_llm_client()
            user_name = user_context.get("username", "there")
            prompt = self.user_prompt_template.format(user_name=user_name)
            try:
                response = await client.chat.completions.create(
                    model=self.llm_model,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt},
                        {"role": "user", "content": user_message}
                    ],
                    temperature=self.llm_temperature,
                    max_tokens=self.llm_max_tokens
                )
                content = response.choices[0].message.content
                try:
                    trace_model_call(
                        provider="openai", model_name=self.llm_model,
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        latency_ms=None,
                        response_summary=content[:200] if content else ""
                    )
                except Exception:
                    pass
                result = {"success": True, "response": content}
                step.capture(result)
                return result
            except Exception as e:
                logger.error(f"LLM chat_response failed: {e}")
                result = {"success": False, "error": str(e)}
                step.capture(result)
                return result

# --- FastAPI App and Endpoints ---

app = FastAPI(
    title="Ecommerce Attendance Tracker Agent",
    description="Multimodal attendance tracking agent for ecommerce HR/employee use.",
    version="1.0.0"
)

# CORS (allow all for demo; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = EcommerceAttendanceTrackerAgent()

# --- Exception Handlers ---

@app.exception_handler(ValidationError)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def validation_exception_handler(request: Request, exc: ValidationError):
    logger.warning(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "success": False,
            "error": "Input validation error",
            "details": exc.errors(),
            "tips": [
                "Check your JSON formatting (quotes, commas, brackets).",
                "Ensure all required fields are present and valid.",
                "Text input must not exceed 50,000 characters."
            ]
        }
    )

@app.exception_handler(HTTPException)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTPException: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "tips": [
                "Check your request and try again.",
                "Contact support if the issue persists."
            ]
        }
    )

@app.exception_handler(Exception)
@with_content_safety(config=GUARDRAILS_CONFIG)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "tips": [
                "Ensure your JSON is well-formed.",
                "Try again later or contact support."
            ]
        }
    )

# --- Endpoints ---

@app.post("/process_input")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def process_input_endpoint(payload: dict):
    """
    Process user input (text or image) and extract intent/entities.
    """
    try:
        input_type = payload.get("input_type")
        if input_type == "text":
            model = TextInputModel(**payload)
        elif input_type == "image":
            model = ImageInputModel(**payload)
        else:
            raise HTTPException(status_code=400, detail="input_type must be 'text' or 'image'.")
        result = await agent.process_input(model.input_data, model.input_type, model.user_context)
        return {"success": True, "result": result}
    except ValidationError as ve:
        raise ve
    except Exception as e:
        logger.error(f"process_input error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tips": [
                "Check your input_type and input_data.",
                "Ensure input_data is valid text or base64-encoded image."
            ]
        }

@app.post("/validate_attendance_entry")
async def validate_attendance_entry_endpoint(payload: dict):
    """
    Validate attendance entry against policy and schedule.
    """
    try:
        model = AttendanceValidationModel(**payload)
        result = await agent.validate_attendance_entry(
            model.employee_id, model.check_in_time, model.shift_start_time, model.input_source
        )
        return {"success": True, "result": result}
    except ValidationError as ve:
        raise ve
    except Exception as e:
        logger.error(f"validate_attendance_entry error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tips": [
                "Check employee_id, check_in_time, shift_start_time, and input_source.",
                "Times must be ISO format strings."
            ]
        }

@app.post("/generate_attendance_report")
async def generate_attendance_report_endpoint(payload: dict):
    """
    Generate attendance report for specified date range.
    """
    try:
        model = ReportGenerationModel(**payload)
        result = await agent.generate_attendance_report(
            model.employee_id, model.date_range, model.report_type
        )
        return result
    except ValidationError as ve:
        raise ve
    except Exception as e:
        logger.error(f"generate_attendance_report error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tips": [
                "Check employee_id, date_range, and report_type.",
                "report_type must be 'daily', 'weekly', or 'monthly'."
            ]
        }

@app.post("/detect_anomalies")
async def detect_anomalies_endpoint(payload: dict):
    """
    Detect suspicious attendance patterns.
    """
    try:
        attendance_records = payload.get("attendance_records")
        if not isinstance(attendance_records, list):
            raise HTTPException(status_code=400, detail="attendance_records must be a list of records.")
        anomalies = await agent.detect_anomalies(attendance_records)
        return {"success": True, "anomalies": anomalies}
    except ValidationError as ve:
        raise ve
    except Exception as e:
        logger.error(f"detect_anomalies error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tips": [
                "attendance_records must be a list of attendance record dicts."
            ]
        }

@app.post("/send_notification")
async def send_notification_endpoint(payload: dict):
    """
    Send notification to user or HR.
    """
    try:
        model = NotificationModel(**payload)
        result = await agent.send_notification(
            model.recipient, model.message, model.notification_type
        )
        return result
    except ValidationError as ve:
        raise ve
    except Exception as e:
        logger.error(f"send_notification error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tips": [
                "Check recipient email, message, and notification_type ('alert' or 'confirmation')."
            ]
        }

@app.post("/authenticate_user")
async def authenticate_user_endpoint(payload: dict):
    """
    Authenticate user via SSO and 2FA.
    """
    try:
        model = UserCredentialsModel(**payload)
        result = await agent.authenticate_user(model.model_dump())
        return result
    except ValidationError as ve:
        raise ve
    except Exception as e:
        logger.error(f"authenticate_user error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tips": [
                "Check username, password, and two_factor_code."
            ]
        }

@app.post("/chat")
@with_content_safety(config=GUARDRAILS_CONFIG)
async def chat_endpoint(payload: dict):
    """
    Chat with the attendance agent (LLM-powered).
    """
    try:
        user_message = payload.get("user_message")
        user_context = payload.get("user_context", {})
        if not user_message or not isinstance(user_message, str):
            raise HTTPException(status_code=400, detail="user_message must be a non-empty string.")
        if len(user_message) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail=f"user_message exceeds {MAX_TEXT_LENGTH} characters.")
        result = await agent.chat_response(user_message, user_context)
        return result
    except ValidationError as ve:
        raise ve
    except Exception as e:
        logger.error(f"chat_endpoint error: {e}")
        return {
            "success": False,
            "error": str(e),
            "tips": [
                "Check user_message and user_context.",
                "user_message must be a non-empty string."
            ]
        }

# --- Main Entrypoint ---



async def _run_with_eval_service():
    """Entrypoint: initialises observability then runs the agent."""
    import logging as _obs_log
    _obs_logger = _obs_log.getLogger(__name__)
    # ── 1. Observability DB schema ─────────────────────────────────────
    try:
        from observability.database.engine import create_obs_database_engine
        from observability.database.base import ObsBase
        import observability.database.models  # noqa: F401 – register ORM models
        _obs_engine = create_obs_database_engine()
        ObsBase.metadata.create_all(bind=_obs_engine, checkfirst=True)
    except Exception as _e:
        _obs_logger.warning('Observability DB init skipped: %s', _e)
    # ── 2. OpenTelemetry tracer ────────────────────────────────────────
    try:
        from observability.instrumentation import initialize_tracer
        initialize_tracer()
    except Exception as _e:
        _obs_logger.warning('Tracer init skipped: %s', _e)
    # ── 3. Evaluation background worker ───────────────────────────────
    _stop_eval = None
    try:
        from observability.evaluation_background_service import (
            start_evaluation_worker as _start_eval,
            stop_evaluation_worker as _stop_eval_fn,
        )
        await _start_eval()
        _stop_eval = _stop_eval_fn
    except Exception as _e:
        _obs_logger.warning('Evaluation worker start skipped: %s', _e)
    # ── 4. Run the agent ───────────────────────────────────────────────
    try:
        import uvicorn
        logger.info("Starting Ecommerce Attendance Tracker Agent on http://0.0.0.0:8000")
        uvicorn.run("agent:app", host="0.0.0.0", port=8000, reload=True)
        pass  # TODO: run your agent here
    finally:
        if _stop_eval is not None:
            try:
                await _stop_eval()
            except Exception:
                pass


if __name__ == "__main__":
    import asyncio as _asyncio
    _asyncio.run(_run_with_eval_service())