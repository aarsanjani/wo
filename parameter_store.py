import json
import os.path

class ParameterStore():
    """
    A simple class for loading and storing JSON documents in a local file.
    """

    def __init__(self, path='', parameters={}, filename='parameters.json'):
        self.filename = path + filename

        if os.path.exists(self.filename):
            self.load()
            self.parameters.update(parameters)

        else:
            self.parameters = parameters
    
    def clear(self):
        self.parameters = {}

    def create(self, parameters={}):
        self.parameters = parameters

    def delete(self, key):
        self.parameters.__delitem__(key)
        
    def read(self):
        return self.parameters

    def update(self, parameters={}):
        self.parameters.update(parameters)

    def load(self):
        with open(self.filename, 'r') as f:
            document = f.read()

        self.parameters = json.loads(document)

    def store(self):
        with open(self.filename, 'w') as f:
            f.write(json.dumps(self.parameters))
