from ultralytics import YOLO
import os

def show_model_labels(model_path):
    try:
        # Get absolute path of the model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, model_path)
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
            
        # Load the model
        model = YOLO(model_path)
        
        # Print the model's labels/classes
        print("\nModel Labels:")
        print("-------------")
        for i, label in enumerate(model.names.values()):
            print(f"{i}: {label}")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    model_path = "D:/demo/best.pt"
    show_model_labels(model_path)
