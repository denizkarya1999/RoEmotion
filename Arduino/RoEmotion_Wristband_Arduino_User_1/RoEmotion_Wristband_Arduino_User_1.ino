// define led according to pin diagram in article
// OOK sign number for User 1 - 10101
const int redLed2 = D2; // there is no LED_BUILTIN available for the XIAO ESP32C3.
const int greenLed4 = D4;

void setup() {
  // initialize digital pin led as an output
  pinMode(redLed2, OUTPUT);
  pinMode(greenLed4, OUTPUT);
}

void loop() {
    digitalWrite(redLed2, LOW);   // 0 turn off the Red Led 
    delayMicroseconds(166);
    digitalWrite(greenLed4, HIGH);   // 1  turns on the Green LED
    delayMicroseconds(166);                
    digitalWrite(greenLed4, LOW);    // 0  turns off the Green LED
    delayMicroseconds(166);
    digitalWrite(greenLed4, HIGH);   // 1  turns on the Green LED
    delayMicroseconds(166);                  
    digitalWrite(greenLed4, LOW);  // 0  turns off the Green LED
    delayMicroseconds(166);
}