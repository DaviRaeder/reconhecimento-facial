import cv2
import mediapipe as mp
import os
import face_recognition
from fer import FER
import random

class FaceRecognition:
    def __init__(self):
        self.webcam = cv2.VideoCapture(0)

        self.reconhecimento_rosto = mp.solutions.face_detection
        self.desenho = mp.solutions.drawing_utils
        self.reconhecedor_rosto = self.reconhecimento_rosto.FaceDetection(min_detection_confidence=0.5)

        self.reconhecimento_mao = mp.solutions.hands
        self.mao_detector = self.reconhecimento_mao.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.detector_emocoes = FER()

        self.image_folder = "rostos_registrados"
        os.makedirs(self.image_folder, exist_ok=True)

    def registrar_rosto(self, frame, nome):
        caminho_imagem = os.path.join(self.image_folder, f"{nome}.jpg")
        cv2.imwrite(caminho_imagem, frame)
        print(f"Rosto registrado: {nome}")

    def reconhecer_rosto(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rosto_desconhecido_encodings = face_recognition.face_encodings(frame_rgb)

        if rosto_desconhecido_encodings:
            rosto_desconhecido_encoding = rosto_desconhecido_encodings[0]
            for filename in os.listdir(self.image_folder):
                caminho_imagem = os.path.join(self.image_folder, filename)
                imagem_registrada = face_recognition.load_image_file(caminho_imagem)
                imagem_registrada_encodings = face_recognition.face_encodings(imagem_registrada)

                if imagem_registrada_encodings:
                    imagem_registrada_encoding = imagem_registrada_encodings[0]
                    resultados = face_recognition.compare_faces([imagem_registrada_encoding], rosto_desconhecido_encoding)

                    if resultados[0]:
                        nome_reconhecido = filename.split(".")[0]
                        cv2.putText(frame, f'Reconhecido: {nome_reconhecido}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        print(f"Rosto reconhecido: {nome_reconhecido}")
                        return nome_reconhecido

        print("Rosto desconhecido")
        return "Desconhecido"

    def detectar_emocao(self, frame):
      if random.randint(1, 6) > 2:
        resultados = self.detector_emocoes.detect_emotions(frame)
        if resultados:
          emocao = resultados[0]["emotions"]
          emocao_principal = max(emocao, key=emocao.get)
          cv2.putText(frame, f'Emocao: {emocao_principal}', (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
          print(f"Emoção detectada: {emocao_principal}")

    def contar_dedos(self, landmarks):
        dedos = 0
        if landmarks[4].x < landmarks[3].x:
            dedos += 1
        for i in range(8, 21, 4):
            if landmarks[i].y < landmarks[i - 2].y:
                dedos += 1
        return dedos

    def detectar_gesto(self, landmarks):
        if (landmarks[4].x < landmarks[3].x and
            landmarks[8].y > landmarks[6].y and
            landmarks[12].y > landmarks[10].y and
            landmarks[16].y > landmarks[14].y and
            landmarks[20].y > landmarks[18].y):
            return "Joinha"
        
        elif (landmarks[8].y < landmarks[6].y and
              landmarks[12].y < landmarks[10].y and
              landmarks[16].y > landmarks[14].y and
              landmarks[20].y > landmarks[18].y):
            return "Paz"
        
        return "Nenhum gesto"


    def recognize(self):
        while self.webcam.isOpened():
            validacao, frame = self.webcam.read()
            if not validacao:
                break

            frame_original = frame.copy()
            imagem_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lista_rostos = self.reconhecedor_rosto.process(imagem_rgb)

            if lista_rostos.detections:
                for rosto in lista_rostos.detections:
                    self.desenho.draw_detection(frame, rosto)

                    if cv2.waitKey(1) & 0xFF == ord('r'):
                        nome = input("Digite o nome do rosto: ")
                        self.registrar_rosto(frame_original, nome)

                    if cv2.waitKey(1) & 0xFF == ord('c'):
                        nome_reconhecido = self.reconhecer_rosto(frame)
                        cv2.putText(frame, f'Reconhecido: {nome_reconhecido}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    self.detectar_emocao(frame)

            lista_maos = self.mao_detector.process(imagem_rgb)

            if lista_maos.multi_hand_landmarks:
                for mao in lista_maos.multi_hand_landmarks:
                    self.desenho.draw_landmarks(frame, mao)
                    dedos_levantados = self.contar_dedos(mao.landmark)
                    cv2.putText(frame, f'Dedos levantados: {dedos_levantados}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    gesto = self.detectar_gesto(mao.landmark)
                    cv2.putText(frame, f'Gesto: {gesto}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Rostos, Gestos e Emoções", frame)

            if cv2.waitKey(5) == 27:
                break

        self.webcam.release()
        cv2.destroyAllWindows()

ReconhecedorDeFaces = FaceRecognition()
ReconhecedorDeFaces.recognize()
