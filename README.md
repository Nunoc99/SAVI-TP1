# SAVI-TP1

Pratical Work 1  - Real Time Face Recognition script <br>
SAVI - 2023/2024 Universidade de Aveiro

## Contribuidores: 
### Grupo 5
- José Silva (103268) josesilva8@ua.pt
- Mário Vasconcelos (84081) mario.vasconcelos@ua.pt
- Nuno Cunha (95167) nunocunha99@ua.pt
  
## Instalação
Dependências de pacotes:
- `pip install mediapipe`
- `pip install face_recognition`
- `pip install dlib==19.22`

## Guia de utilizador
(manual)

## Atualizações
Update, o new_main e o faceRecog para correr devem instalar/atualizar o seguinte:

Neste momento, o new_main, dei-lhe este nome pq é temporário, faz o reconhecimento através de uma base de dados.
Ele vai buscar a uma pasta local, que por acaso dei-lhe o nome de "faces" e coloquei uma imagem minha e uma do Messi,
assim sendo, fez bem o reconhecimento com uma accuracy média de 85/90%. Avaliem e digam o que acham por favor.

-- Update 27/10 --
 - Foi feito um ficheiro novo para reconhecimento do rosto, foi utilizada uma libraria que garante ter mais fps e aumentar a fluidez do vídeo, que está incorporada no modo de funcionamento do programa CDM.
 - Ou seja, o CDM funciona na perfeição e tira tanto fotos frontais se carregarmos no "f", como de perfil se carregarmos no "l", "f" de frente e "l" de lateral.
 - O modo FDM funciona também muito bem, fica mais lento mas também tem mais processamento. 
 - Quanto ao ficheiro faceRecog, fiz umas pequenas alterações para melhorar a comparação e deteção. 

Deste modo, fica a faltar aplicar o tracking e dps se tivermos algum tempo, aplicar alguns extras e o text-to-speech





