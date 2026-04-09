import speech_recognition as sr
import keyboard
import time

r = sr.Recognizer()
mic = sr.Microphone()


while True:
    if keyboard.is_pressed('g'):
        with mic as source:
            print("\n พูด 3 วินาที")
            audio = r.listen(source, phrase_time_limit=3)
        try:
            text = r.recognize_google(audio, language='th-TH')
            print("ข้อความ: " + text)
        except:
            print("ผิดพลาด ลองใหม่")

        time.sleep(1)
        print("\nกด 'g' เพื่อพูดใหม่")
    time.sleep(0.1)