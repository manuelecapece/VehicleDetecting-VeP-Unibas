Nel progetto realizzato si vuole proporre un sistema di rilevamento e conteggio di veicoli in transito su un'autostrada, in varie condizioni metereologiche.
Il sistema proposto fa uso di Yolov8 per il rilevamento dei veicoli autostradali ripresi da un cavalcavia, 
ByteTrack per il loro tracciamento e l'ultima libreria Python di Roboflow - Supervision, per il conteggio dei veicoli.

All'interno della cartella supervision si trovano le classi che abbiamo modificato e quelle che abbiamo aggiunto in base alle nostre necessità.

ISTRUZIONI ESECUZIONE DEL CODICE
Il progetto è stato realizzato attraverso un notebook Colab nominato VeichleCounting, situato nella cartella Notebook sia su gitHub che sulla cartella drive del progetto.
1) Nella prima sezione vengono installati tutti gli strumenti necessari.
2) Successivamente vengono scaricati all'interno del notebook i video su cui è possibile effettuare le predizioni
  In questa fase viene scelto il video su cui eseguire il processo.
3)Conteggio e rilevamento
  In questa fase vengono generati corrispettivamente due video. 
  Il primo in cui il software conterà indistintivamente ogni veicolo che transita dalla linea di demarcazione.
  Il secondo in cui vengono contati i veicoli transitanti rispettivamente per ogni categoria.
Una volta generati i video vengono caricati sulla cartella drive del progetto, il link che porta alla loro posizione viene mostrato nel notebook.
