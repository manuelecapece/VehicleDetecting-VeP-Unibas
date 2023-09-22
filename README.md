# ITA
Nel progetto realizzato si vuole proporre un sistema di rilevamento e conteggio di veicoli in transito su un'autostrada, in varie condizioni metereologiche.
Il sistema proposto fa uso di Yolov8 per il rilevamento dei veicoli autostradali ripresi da un cavalcavia, 
ByteTrack per il loro tracciamento e l'ultima libreria Python di Roboflow - Supervision, per il conteggio dei veicoli.

All'interno della cartella supervision si trovano le classi che abbiamo modificato e quelle che abbiamo aggiunto in base alle nostre necessità.

## ISTRUZIONI ESECUZIONE DEL CODICE

Il progetto è stato realizzato attraverso un notebook Colab nominato VeichleCounting, situato nella cartella Notebook sia su gitHub che sulla cartella drive del progetto (vedi VeP-progetto per i riferimenti).
### 1) Nella prima sezione vengono installati tutti gli strumenti necessari.
  - Inanzitutto si procede ad installare la versione 8.0.20 di Ultralytics (Yolov8)
  - Nella cella successiva si trova il codice per installare Bytetrack
  - L'ultima libreria da installare è Supervision
  - Infine viene caricato il modello pre-addestrato XtraLarge di yolo ovvero: yolov8x.pt
### 2) Successivamente vengono scaricati all'interno del notebook i video su cui è possibile effettuare le predizioni.
  In questa fase viene scelto il video su cui eseguire il processo. Inoltre è possibile anche cambiare i punti di LINE_START e di LINE_END conservati nel dizionario.
  I punti si differenziano per ogni video in modo tale da adattare il counting ad ogni scena.
### 3) Conteggio e rilevamento
  In questa fase vengono generati rispettivamente due video. 
  Il primo in cui il software conterà indistintivamente ogni veicolo che transita dalla linea di demarcazione.
  Il secondo in cui vengono contati i veicoli transitanti rispettivamente per ogni categoria.
Una volta generati i video vengono caricati sulla cartella drive del progetto, il link che porta alla loro posizione viene mostrato nel notebook.

# ENG
In the project we aim to propose a system for the detection and counting of vehicles traveling on a highway under various weather conditions.
The proposed system uses Yolov8 for the detection of highway vehicles captured from an overpass, ByteTrack for their tracking, and Roboflow's latest Python library - Supervision, for vehicle counting.

Inside the "supervision" folder, you can find the classes we modified and added based on our needs.

## CODE EXECUTION INSTRUCTIONS

The project was implemented through a Colab notebook named "VehicleCounting," located in both the GitHub repository and the project's Google Drive folder (see "VeP-progetto" for references).

### 1) In the first section, all necessary tools are installed.
  - First, we install version 8.0.20 of Ultralytics (Yolov8).
  - In the next cell, you'll find the code to install ByteTrack.
  - The last library to install is Supervision.
  - Finally, we load the pre-trained XtraLarge YOLO model, namely "yolov8x.pt."

### 2) Next, the videos on which predictions can be made are downloaded within the notebook.
  In this phase, you choose the video on which to run the process. It is also possible to change the LINE_START and LINE_END points stored in the dictionary.
  These points differ for each video, adapting the counting to each scene.

### 3) Counting and Detection
  In this phase, two videos are generated:
  - The first video where the software will indiscriminately count every vehicle passing the demarcation line.
  - The second video where the passing vehicles are counted by category.
Once the videos are generated, they are uploaded to the project's Google Drive folder, and the link leading to their location is shown in the notebook.
