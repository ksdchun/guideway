// Define motor control pins
const int ENA = 10;  // PWM pin for speed control
const int IN1 = 8;   // Direction control pin 1
const int IN2 = 9;   // Direction control pin 2

char command = '0'; // Default state: motor running

void setup() {
    pinMode(ENA, OUTPUT);
    pinMode(IN1, OUTPUT);
    pinMode(IN2, OUTPUT);

    Serial.begin(9600);
}


void loop() {
    // Check if a command is available from Python
    if (Serial.available() > 0) {
        command = Serial.read();
        Serial.println("Received: " + String(command)); // Debugging output
    }

    // Run motor at full power when "1", stop when "0"
    if (command == '1') {
        driveMotor(255, 1);  // Full speed forward
    } else if (command == '0') {
        driveMotor(0, 1);     // Stop motor
    }
}

void driveMotor(int speed, bool direction) {
    analogWrite(ENA, speed);
    if (direction) {
        digitalWrite(IN1, HIGH);
        digitalWrite(IN2, LOW);
    } else {
        digitalWrite(IN1, LOW);
        digitalWrite(IN2, HIGH);
    }
}
