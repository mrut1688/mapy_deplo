const int eeg[]={34,35,32,33,25,26,27};

void setup() {
  Serial.begin(115200);
}

void loop() {
Serial.print(EEGFilter(analogRead(eeg[4])*3.3000/4096),6);  //T7
Serial.print(",");
Serial.print(EEGFilter(analogRead(eeg[5])*3.3000/4096),6);  //C3
Serial.print(",");
Serial.print(EEGFilter(analogRead(eeg[6])*3.3000/4096),6);  //T8
Serial.print(",");
Serial.print(EEGFilter(analogRead(eeg[7])*3.3000/4096),6);  //C4
Serial.print(",");
Serial.println("");
delay(1);

 }
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
