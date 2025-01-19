import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from deepface import DeepFace
from queue import Queue
from retinaface import RetinaFace

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
    
    def forward(self, x):
        return self.network(x)

class FaceRecognitionSystem:
    def __init__(self, dataset_path, model_name="Facenet512", detector_backend='retinaface', skip_frames=2):
        self.dataset_path = dataset_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.embeddings_db = {}
        self.face_detector = RetinaFace
        self.model = CNNModel()
        self.model.eval()
        self.skip_frames = skip_frames
        self.frame_count = 0
        self.last_results = []
        self.processing_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.is_processing = False
        
    def preprocess_face(self, img, target_size=(112, 112)):
        """Preprocess face image manually."""
        if img is None:
            return None
            
        if img.shape[0] > 0 and img.shape[1] > 0:
            factor_0 = target_size[0] / img.shape[0]
            factor_1 = target_size[1] / img.shape[1]
            factor = min(factor_0, factor_1)
            
            dsize = (int(img.shape[1] * factor), int(img.shape[0] * factor))
            img = cv2.resize(img, dsize)
            
            diff_0 = target_size[0] - img.shape[0]
            diff_1 = target_size[1] - img.shape[1]
            img = cv2.copyMakeBorder(
                img, 
                diff_0 // 2, diff_0 - (diff_0 // 2),
                diff_1 // 2, diff_1 - (diff_1 // 2),
                cv2.BORDER_CONSTANT
            )
            
        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)
            
        if len(img.shape) == 3:
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            img = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
        return img

    def get_embedding(self, image):
        """Extract embedding dengan multiple augmentations."""
        try:
            embeddings = []
            
            img_processed = self.preprocess_face(image)
            if img_processed is not None:
                temp_path = "temp_image.jpg"
                cv2.imwrite(temp_path, img_processed)
                
                embedding = DeepFace.represent(
                    img_path=temp_path,
                    model_name=self.model_name,
                    enforce_detection=False,
                    detector_backend=self.detector_backend
                )
                
                os.remove(temp_path)
                
                if embedding:
                    embeddings.append(embedding[0]["embedding"])
            
            augmentations = [
                lambda x: cv2.flip(x, 1),  
                lambda x: cv2.GaussianBlur(x, (5,5), 0),  
                lambda x: cv2.addWeighted(x, 1.5, x, 0, -50),  
            ]
            
            for aug_func in augmentations:
                try:
                    aug_image = aug_func(image.copy())
                    aug_processed = self.preprocess_face(aug_image)
                    
                    if aug_processed is not None:
                        temp_path = "temp_aug_image.jpg"
                        cv2.imwrite(temp_path, aug_processed)
                        
                        aug_embedding = DeepFace.represent(
                            img_path=temp_path,
                            model_name=self.model_name,
                            enforce_detection=False,
                            detector_backend=self.detector_backend
                        )
                        
                        os.remove(temp_path)
                        
                        if aug_embedding:
                            embeddings.append(aug_embedding[0]["embedding"])
                except Exception as e:
                    print(f"Augmentation error: {e}")
                    continue
            
            if embeddings:
                combined_embedding = np.mean(embeddings, axis=0)
                embedding_tensor = torch.tensor(combined_embedding).float()
                
                with torch.no_grad():
                    processed_embedding = self.model(embedding_tensor.unsqueeze(0))
                
                return processed_embedding.squeeze(0).numpy()
            
            return None
            
        except Exception as e:
            print(f"Error generating embedding: {e}")
            return None

    def build_database(self):
        """Build database dengan augmented embeddings."""
        print("Building face database...")
        for person_name in os.listdir(self.dataset_path):
            person_folder = os.path.join(self.dataset_path, person_name)
            if os.path.isdir(person_folder):
                self.embeddings_db[person_name] = []
                image_files = [f for f in os.listdir(person_folder) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                for img_name in image_files:
                    img_path = os.path.join(person_folder, img_name)
                    try:
                        img = cv2.imread(img_path)
                        if img is not None:
                            embedding = self.get_embedding(img)
                            if embedding is not None:
                                self.embeddings_db[person_name].append(embedding)
                                print(f"Processed {person_name}")
                    except Exception as e:
                        print(f"Error processing {img_name}: {str(e)}")

    def find_closest_match(self, query_embedding, threshold=0.4):
        """Find match dengan ensemble voting."""
        distances = {}
        query_tensor = torch.tensor(query_embedding).float()
        
        for person_name, embeddings in self.embeddings_db.items():
            person_distances = []
            for ref_embedding in embeddings:
                ref_tensor = torch.tensor(ref_embedding).float()
                
                cosine_dist = torch.nn.functional.cosine_similarity(
                    query_tensor.unsqueeze(0),
                    ref_tensor.unsqueeze(0)
                ).item()
                
                euclidean_dist = torch.nn.functional.pairwise_distance(
                    query_tensor.unsqueeze(0),
                    ref_tensor.unsqueeze(0)
                ).item()
                
                # Combine distances
                combined_dist = (cosine_dist + (1 - euclidean_dist)) / 2
                person_distances.append(combined_dist)
            
            if person_distances:
                distances[person_name] = np.mean(person_distances)
        
        if distances:
            best_match = max(distances.items(), key=lambda x: x[1])
            similarity = max(0, min(100, best_match[1] * 100))
            
            if similarity < threshold * 100:
                return None, 0
            return best_match[0], similarity
        
        return None, 0

    def process_frame_async(self):
        """Process frames asynchronously dengan enhanced detection."""
        while self.is_processing:
            if not self.processing_queue.empty():
                frame = self.processing_queue.get()
                
                try:
                    faces = self.face_detector.detect_faces(frame)
                    results = []
                    
                    if isinstance(faces, dict):
                        for face_idx in faces.keys():
                            face_data = faces[face_idx]
                            facial_area = face_data['facial_area']
                            
                            x, y = facial_area[0], facial_area[1]
                            w = facial_area[2] - facial_area[0]
                            h = facial_area[3] - facial_area[1]
                            
                            margin = 0.2
                            x1 = max(0, int(x - margin * w))
                            y1 = max(0, int(y - margin * h))
                            x2 = min(frame.shape[1], int(x + w + margin * w))
                            y2 = min(frame.shape[0], int(y + h + margin * h))
                            
                            face_img = frame[y1:y2, x1:x2]
                            
                            if face_img.size > 0:
                                embedding = self.get_embedding(face_img)
                                if embedding is not None:
                                    person, confidence = self.find_closest_match(embedding)
                                    results.append({
                                        'bbox': (x1, y1, x2-x1, y2-y1),
                                        'person': person,
                                        'confidence': confidence
                                    })
                    
                    if not self.result_queue.full():
                        self.result_queue.put(results)
                        
                except Exception as e:
                    print(f"Error in frame processing: {e}")
                    if not self.result_queue.full():
                        self.result_queue.put([])