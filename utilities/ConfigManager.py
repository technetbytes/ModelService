from assets import file_storage

class FileConfigManager:
    __shared_instance = 'NONE'
  
    @staticmethod
    def get_Instance():
        """Static Access Method"""
        if ConfigManager.__shared_instance == 'NONE': 
            ConfigManager() 
        return ConfigManager.__shared_instance 
  
    def __init__(self):
        print(ConfigManager.__shared_instance)   
        if ConfigManager.__shared_instance == 'NONE': 
            ConfigManager.__shared_instance = self
    
    def get_FileManager(self,storage_type=None, config):
        file_manager = file_storage.FileManager(storage_type)
        file_manager.load_configuration(config)
        return file_manager
