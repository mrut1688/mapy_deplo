#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>

const int eeg[] = {34, 35, 32, 33, 25, 26, 27};

BLEServer *pServer;
BLECharacteristic *pCharacteristic;
bool deviceConnected = false;
bool oldDeviceConnected = false;

void setup() {
  Serial.begin(115200);

  
  BLEDevice::init("ESP32_BLE_Server");
  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  
  BLEService *pService = pServer->createService(BLEUUID("0000180d-0000-1000-8000-00805f9b34fb"));

  
  pCharacteristic = pService->createCharacteristic(
      BLEUUID("00002a37-0000-1000-8000-00805f9b34fb"),
      BLECharacteristic::PROPERTY_READ
  );

  pService->start();

  
  pServer->getAdvertising()->start();
  Serial.println("Waiting for a connection...");
}

void loop() {
  if (deviceConnected) 
  {
    
    String eegData = "";
    
    
    eegData = String(EEGFilter(analogRead(eeg[0]) * 3.3000 / 4096), 6);    
    pCharacteristic->setValue(eegData.c_str());
    eegData =","
    pCharacteristic->setValue(eegData.c_str());
    eegData = String(EEGFilter(analogRead(eeg[1]) * 3.3000 / 4096), 6);    
    pCharacteristic->setValue(eegData.c_str());
    eegData =","
    pCharacteristic->setValue(eegData.c_str());
    eegData = String(EEGFilter(analogRead(eeg[2]) * 3.3000 / 4096), 6);    
    pCharacteristic->setValue(eegData.c_str());
    eegData =","
    pCharacteristic->setValue(eegData.c_str());
    eegData = String(EEGFilter(analogRead(eeg[3]) * 3.3000 / 4096), 6);    
    pCharacteristic->setValue(eegData.c_str());
    eegData ="\n"
    pCharacteristic->setValue(eegData.c_str());
    delay(0.1); 
  }


  if (!deviceConnected && oldDeviceConnected) {
    delay(500); 
    pServer->startAdvertising(); 
    Serial.println("Waiting for a connection...");
    oldDeviceConnected = deviceConnected;
  }

  
  if (deviceConnected != oldDeviceConnected) {
    if (deviceConnected) {
      Serial.println("Connected");
    } else {
      Serial.println("Disconnected");
    }
    oldDeviceConnected = deviceConnected;
  }
}

class MyServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer* pServer) {
    deviceConnected = true;
  };

  void onDisconnect(BLEServer* pServer) {
    deviceConnected = false;
  }
};

float EEGFilter(float input) {
	float output = input;
	{
		static float z1, z2; // filter section state
		float x = output - -0.95391350*z1 - 0.25311356*z2;
		output = 0.00735282*x + 0.01470564*z1 + 0.00735282*z2;
		z2 = z1;
		z1 = x;
	}
	{
		static float z1, z2; // filter section state
		float x = output - -1.20596630*z1 - 0.60558332*z2;
		output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
		z2 = z1;
		z1 = x;
	}
	{
		static float z1, z2; // filter section state
		float x = output - -1.97690645*z1 - 0.97706395*z2;
		output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
		z2 = z1;
		z1 = x;
	}
	{
		static float z1, z2; // filter section state
		float x = output - -1.99071687*z1 - 0.99086813*z2;
		output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
		z2 = z1;
		z1 = x;
	}
	return output;
}

