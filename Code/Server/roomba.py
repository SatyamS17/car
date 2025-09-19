from time import sleep
from ultrasonic import Ultrasonic
from motor import Ordinary_Car
import random

if __name__ == '__main__':
    ultrasonic = Ultrasonic()
    PWM = Ordinary_Car()

    # PWM.set_motor_model(2000,2000,-2000,-2000) 

    try:
        while True:
            dist = ultrasonic.get_distance()
            print(f"Distance: {dist:.1f} cm")

            if dist <= 30.0:
                print("STOPPING")
                PWM.set_motor_model(0, 0, 0, 0)
                sleep(1)
                print("STOPPING DONE")

                print("BACKING UP")
                PWM.set_motor_model(-1000, -1000, -1000, -1000)
                sleep(0.5)
                print("BACKING DONE")

                print("TURNING")
                turnPower = random.choice([-1000, 1000])
                PWM.set_motor_model(turnPower, turnPower, -turnPower, -turnPower)

                sleep(2)
                print("TURNING DONE")

                # while ultrasonic.get_distance() <= 30:
                #     sleep(0.1)

                print("STOPPING 2nd")
                PWM.set_motor_model(0, 0, 0, 0)
                sleep(0.5)
                print("STOPPING 2nd DONE")

            else:
                print("MOVING FORWARD")
                PWM.set_motor_model(1000, 1000, 1000, 1000)
                sleep(0.1)
                print("FORWARD DONE")

    finally:
        # This always runs when the program exits
        PWM.set_motor_model(0, 0, 0, 0)
        print("Car stopped!")
