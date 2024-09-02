const int stepPin = 3;
const int dirPin = 4;

void setup() {
  Serial.begin(9600);
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
}

void loop() {
  if (Serial.available() > 0) {
    String command = Serial.readStringUntil('\n');
    if (command.startsWith("MOVE ")) {
      int steps = command.substring(5).toInt();
      moveMotor(steps);
      Serial.println("DONE");
    }
  }
}

void moveMotor(int steps) {
  digitalWrite(dirPin, steps > 0 ? HIGH : LOW);
  steps = abs(steps);
  for (int i = 0; i < steps; i++) {
    digitalWrite(stepPin, HIGH);
    delayMicroseconds(500);
    digitalWrite(stepPin, LOW);
    delayMicroseconds(500);
  }
}