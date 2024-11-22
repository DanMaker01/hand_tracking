
'''
0  - base da mão
1  - 
2  - 
3  - 
4  - polegar ponta
5  - 
6  - 
7  - 
8  - indicador ponta
9  - 
10 - 
11 - 
12 - meio ponta
13 - 
14 - 
15 - 
16 - anelar ponta
17 - 
18 - 
19 - 
20 - mindinho ponta
'''
LAR = 800
ALT = 600

import math
import cv2
import mediapipe as mp

# Importar o mediaPipe para reconhecimento de mãos
# e para desenhar os resultados
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Abrir a webcam para capturar o vídeo
# Isso criar  um objeto de captura de vídeo
# que pode ser usado para ler frames do vídeo
# da webcam
cap = cv2.VideoCapture(0)

# Definir o nome da janela e permitir redimensionamento
cv2.namedWindow('Hand Tracking', cv2.WINDOW_NORMAL)

# Definir o tamanho desejado para a janela (largura, altura)
cv2.resizeWindow('Hand Tracking', LAR,ALT)


# Configurar o MediaPipe para reconhecer as maos
# Isso criar  um objeto que pode ser usado
# para processar frames do vídeo e detectar as mãos
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    # Enquanto a captura de vídeo estiver aberta
    while cap.isOpened():
        # Ler um frame do vídeo
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Inverter a imagem horizontalmente
        # isso é necessário para que a imagem seja
        # exibida como uma selfie
        image = cv2.flip(image, 1)

        # Converter a imagem de BGR para RGB
        # isso é necessário para que o MediaPipe possa
        # processar a imagem
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Processar a imagem com o MediaPipe
        results = hands.process(image_rgb)

        # for j in range(0,int(0.8*LAR),int(0.8*LAR/10)):
        #     for k in range(0,int(0.8*ALT),int(0.8*ALT/10)):
        #         cv2.putText(image, f'{j},{k}', (j,k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        # Se houver alguma mão detectada
        if results.multi_hand_landmarks:
            # Iterar sobre as mãos detectadas
            for hand_landmarks in results.multi_hand_landmarks:
                # Desenhar as marcas da mão na imagem
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # mostrar texto na tela
                #transformar discionario em lista
                hand_landmarks = list(hand_landmarks.landmark)
                # print(hand_landmarks)
                # desenha a lista de marcadores com posição (x,y,z)
                
                
                for i, landmark in enumerate(hand_landmarks):
                    x = landmark.x
                    y = landmark.y
                    z = landmark.z
                    XX = math.floor(x*LAR)
                    YY = math.floor(y*ALT)
                    # cv2.putText(image, f'{i}', (XX,YY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 2)
                    cv2.putText(image, f'{i}: X: {x:.2f} Y: {y:.2f} Z: {z:.2f}', (10, 30 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
                # for i, landmark in enumerate(hand_landmarks):
                #     x = landmark.x
                #     y = landmark.y
                #     z = landmark.z
                #     cv2.putText(image, f'{i}: X: {x:.2f} Y: {y:.2f} Z: {z:.2f}', (10, 30 + (i * 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
                
        # Exibir a imagem
        cv2.imshow('Hand Tracking', image)

        # Sair do loop quando a tecla 'q' for pressionada
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

# Fechar a captura de vídeo
# Isso liberar  recursos do sistema
cap.release()
# Fechar todas as janelas abertas pelo OpenCV
# Isso liberar  recursos do sistema
cv2.destroyAllWindows()

