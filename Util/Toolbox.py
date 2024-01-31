import os


#AUX
def generate_model_name()->str:
        import uuid
        return uuid.uuid4()

class Toolbox:
    def __init__(self,model_name = generate_model_name()) -> None:
        self.model_name = model_name
    
    # Generate a unique model name
    def generate_model_name(self)->str:
        return generate_model_name()
    
    def log(event):
        from datetime import datetime
        with open('filename.txt', 'a') as f:
            f.write(f'[{datetime.now}]{event}\n')
