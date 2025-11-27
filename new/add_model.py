from model_manager import model_manager

model_manager.register_model(
    model_name="gemini-2.5-pro",
    provider="google"
)

print(model_manager.models)