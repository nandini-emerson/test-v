
import os
import logging
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# --- Logging Setup ---
logger = logging.getLogger("attendance_config")
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# --- Default Values ---
DEFAULT_LLM_CONFIG = {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 2000,
    "system_prompt": (
        "You are the Ecommerce Attendance Tracker Agent. Your role is to help employees and HR staff track, validate, "
        "and report attendance in a friendly, accurate, and policy-compliant manner. Always verify identity, follow business rules, "
        "and escalate issues as needed."
    ),
    "user_prompt_template": "Hi {user_name}, how can I assist you with your attendance today?",
    "few_shot_examples": [
        "I want to check in for my shift.",
        "Why was my attendance marked as late yesterday?"
    ]
}

DEFAULT_DOMAIN = "ecommerce"
DEFAULT_AGENT_NAME = "Ecommerce Attendance Tracker Agent edited"

# --- API Key Management & Validation ---
def get_env_var(key, required=True, fallback=None):
    value = os.getenv(key, fallback)
    if required and not value:
        logger.error(f"Missing required environment variable: {key}")
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value

class APISettings:
    # External APIs
    HRIS_API_URL = get_env_var("HRIS_API_URL", required=True)
    HRIS_API_AUTH = get_env_var("HRIS_API_AUTH", required=True)  # OAuth2/SSO token
    FACE_RECOGNITION_API_URL = get_env_var("FACE_RECOGNITION_API_URL", required=True)
    FACE_RECOGNITION_API_KEY = get_env_var("FACE_RECOGNITION_API_KEY", required=True)
    EMAIL_NOTIFICATION_API_URL = get_env_var("EMAIL_NOTIFICATION_API_URL", required=True)
    EMAIL_NOTIFICATION_API_KEY = get_env_var("EMAIL_NOTIFICATION_API_KEY", required=True)
    # Internal APIs
    SHIFT_SCHEDULER_API_URL = get_env_var("SHIFT_SCHEDULER_API_URL", required=True)
    SHIFT_SCHEDULER_TOKEN = get_env_var("SHIFT_SCHEDULER_TOKEN", required=True)

    # Rate limits
    RATE_LIMITS = {
        "HRIS_API": 1000,  # requests/hour
        "FaceRecognition": 500,
        "EmailNotification": 1000,
        "ShiftScheduler": None  # Unlimited (internal)
    }

# --- LLM Configuration ---
class LLMConfig:
    PROVIDER = DEFAULT_LLM_CONFIG["provider"]
    MODEL = DEFAULT_LLM_CONFIG["model"]
    TEMPERATURE = DEFAULT_LLM_CONFIG["temperature"]
    MAX_TOKENS = DEFAULT_LLM_CONFIG["max_tokens"]
    SYSTEM_PROMPT = DEFAULT_LLM_CONFIG["system_prompt"]
    USER_PROMPT_TEMPLATE = DEFAULT_LLM_CONFIG["user_prompt_template"]
    FEW_SHOT_EXAMPLES = DEFAULT_LLM_CONFIG["few_shot_examples"]
    OPENAI_API_KEY = get_env_var("OPENAI_API_KEY", required=True)

# --- Domain-Specific Settings ---
class AttendancePolicy:
    CHECKIN_GRACE_MINUTES = int(os.getenv("CHECKIN_GRACE_MINUTES", "15"))
    ABSENCE_REASON_REQUIRED = True
    AUTHORIZED_MODIFIERS = os.getenv("AUTHORIZED_MODIFIERS", "HR,Manager").split(",")
    DATA_RETENTION_DAYS = int(os.getenv("DATA_RETENTION_DAYS", "365"))
    PII_MASKING = True

# --- Validation & Error Handling ---
class ConfigValidator:
    @staticmethod
    def validate():
        errors = []
        # Check all required API keys
        required_keys = [
            "HRIS_API_URL", "HRIS_API_AUTH",
            "FACE_RECOGNITION_API_URL", "FACE_RECOGNITION_API_KEY",
            "EMAIL_NOTIFICATION_API_URL", "EMAIL_NOTIFICATION_API_KEY",
            "SHIFT_SCHEDULER_API_URL", "SHIFT_SCHEDULER_TOKEN",
            "OPENAI_API_KEY"
        ]
        for key in required_keys:
            if not os.getenv(key):
                errors.append(key)
        if errors:
            logger.error(f"Missing required API keys or URLs: {errors}")
            raise RuntimeError(f"Missing required API keys or URLs: {errors}")
        return True

# --- Fallbacks ---
def get_fallback(key):
    fallbacks = {
        "CHECKIN_GRACE_MINUTES": 15,
        "DATA_RETENTION_DAYS": 365,
        "AUTHORIZED_MODIFIERS": ["HR", "Manager"]
    }
    return fallbacks.get(key)

# --- Exported Config Object ---
class AgentConfig:
    # Agent identity
    AGENT_NAME = DEFAULT_AGENT_NAME
    DOMAIN = DEFAULT_DOMAIN

    # API settings
    HRIS_API_URL = APISettings.HRIS_API_URL
    HRIS_API_AUTH = APISettings.HRIS_API_AUTH
    FACE_RECOGNITION_API_URL = APISettings.FACE_RECOGNITION_API_URL
    FACE_RECOGNITION_API_KEY = APISettings.FACE_RECOGNITION_API_KEY
    EMAIL_NOTIFICATION_API_URL = APISettings.EMAIL_NOTIFICATION_API_URL
    EMAIL_NOTIFICATION_API_KEY = APISettings.EMAIL_NOTIFICATION_API_KEY
    SHIFT_SCHEDULER_API_URL = APISettings.SHIFT_SCHEDULER_API_URL
    SHIFT_SCHEDULER_TOKEN = APISettings.SHIFT_SCHEDULER_TOKEN
    RATE_LIMITS = APISettings.RATE_LIMITS

    # LLM
    LLM_PROVIDER = LLMConfig.PROVIDER
    LLM_MODEL = LLMConfig.MODEL
    LLM_TEMPERATURE = LLMConfig.TEMPERATURE
    LLM_MAX_TOKENS = LLMConfig.MAX_TOKENS
    LLM_SYSTEM_PROMPT = LLMConfig.SYSTEM_PROMPT
    LLM_USER_PROMPT_TEMPLATE = LLMConfig.USER_PROMPT_TEMPLATE
    LLM_FEW_SHOT_EXAMPLES = LLMConfig.FEW_SHOT_EXAMPLES
    OPENAI_API_KEY = LLMConfig.OPENAI_API_KEY

    # Attendance policy
    CHECKIN_GRACE_MINUTES = AttendancePolicy.CHECKIN_GRACE_MINUTES
    ABSENCE_REASON_REQUIRED = AttendancePolicy.ABSENCE_REASON_REQUIRED
    AUTHORIZED_MODIFIERS = AttendancePolicy.AUTHORIZED_MODIFIERS
    DATA_RETENTION_DAYS = AttendancePolicy.DATA_RETENTION_DAYS
    PII_MASKING = AttendancePolicy.PII_MASKING

    # Validation
    @classmethod
    def validate(cls):
        return ConfigValidator.validate()

# Validate config on import
try:
    AgentConfig.validate()
except Exception as e:
    logger.error(f"AgentConfig validation failed: {e}")
    raise

# Usage example (import this config in your agent modules):
# from config import AgentConfig
# api_url = AgentConfig.HRIS_API_URL

