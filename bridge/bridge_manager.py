from bridge.databse_bridge import DbBridge

class BridgeManager:
    __shared_instance = 'NONE'
  
    @staticmethod
    def get_Instance():
        """Static Access Method"""
        if BridgeManager.__shared_instance == 'NONE': 
            BridgeManager() 
        return BridgeManager.__shared_instance 
  
    def __init__(self):
        print(BridgeManager.__shared_instance)   
        """virtual private constructor"""
        #if BridgeManager.__shared_instance != 'NONE': 
        #    raise Exception ("This class is a singleton class !") 
        #else: 
        #    BridgeManager.__shared_instance = self
        if BridgeManager.__shared_instance == 'NONE': 
            BridgeManager.__shared_instance = self
    
    def get_Bridge(self):
        bridge = DbBridge()
        bridge.load_db()
        return bridge
