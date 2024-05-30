import torch
import cv2

# Carica il modello YOLOv5 pre-addestrato
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_dog(frame):
    # Esegui l'inferenza con il modello YOLOv5
    results = model(frame)
    # Converti i risultati in un DataFrame Pandas
    results_df = results.pandas().xyxy[0]
    # Filtra i risultati per mantenere solo l'animale specificato
    dog = results_df[results_df['class'] == 16]
    return dog

def detect_cat(frame):
    # Esegui l'inferenza con il modello YOLOv5
    results = model(frame)
    # Converti i risultati in un DataFrame Pandas
    results_df = results.pandas().xyxy[0]
    # Filtra i risultati per mantenere solo l'animale specificato
    cat = results_df[results_df['class'] == 15]
    return cat

def detect_human(frame):
    # Esegui l'inferenza con il modello YOLOv5
    results = model(frame)
    # Converti i risultati in un DataFrame Pandas
    results_df = results.pandas().xyxy[0]
    # Filtra i risultati per mantenere solo l'animale specificato
    human = results_df[results_df['class'] == 0]
    return human

def detect_car(frame):
    # Esegui l'inferenza con il modello YOLOv5
    results = model(frame)
    # Converti i risultati in un DataFrame Pandas
    results_df = results.pandas().xyxy[0]
    # Filtra i risultati per mantenere solo l'animale specificato
    car = results_df[results_df['class'] == 2]
    return car

# Carica il video
video_path = 'humans_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    '''
    dogs = detect_dog(frame)
    cats = detect_cat(frame)
    cars = detect_car(frame)
    '''
    '''
    for _, row in cats.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Cat {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    for _, row in dogs.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Animal {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
    for _, row in dogs.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Animal {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
    
    '''

    humans = detect_human(frame)
    for _, row in humans.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        #confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(frame, 'Human', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    dogs = detect_dog(frame)
    for _, row in dogs.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        #confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Dog', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cats = detect_cat(frame)
    for _, row in cats.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        #confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, 'Cat', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cars = detect_car(frame)
    for _, row in cars.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence = row['confidence']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, 'Car', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow('Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()