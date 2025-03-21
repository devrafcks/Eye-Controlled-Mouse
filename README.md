# Eye Controlled Mouse

Este projeto utiliza a biblioteca `MediaPipe` para detectar os pontos faciais e controlar o movimento do mouse com os olhos. Ele também simula cliques com o piscar de olhos.

## Funcionalidades

- **Controle do mouse:** Move o cursor com base no movimento dos olhos.
- **Cliques com os olhos:** Um piscar de olhos é detectado para simular um clique.
- **Ajuste de sensibilidade:** O movimento do mouse pode ser suavizado e a velocidade ajustada.

## Requisitos

- Python 3.6+
- Bibliotecas:
  - OpenCV
  - MediaPipe
  - PyAutoGUI

## Instalação

Instale as dependências com o comando:

pip install opencv-python mediapipe pyautogui


## Como Usar

1. Execute o script Python.
2. Olhe para a câmera e mova os olhos para controlar o cursor.
3. Pisque os olhos para simular um clique.

## Observações

- O fator de velocidade do cursor pode ser ajustado modificando o valor de `speed_factor`.
- A tolerância para o piscar de olhos pode ser ajustada modificando o valor de comparação em `(left[0].y - left[1].y)`.
