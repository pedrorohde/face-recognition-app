import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, ClientSettings
import face_recognition


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        super().__init__()
        self.process_this_frame = True
        self.face_locations = []
        self.face_embeddings = []
        self.face_names = []
        self.names = np.load('names.npy')
        self.embeddings = np.load('embeddings.npy')
        print(self.embeddings.shape)

    def transform(self, frame):

        img = frame.to_ndarray(format="bgr24")

        small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

        rgb_small_frame = small_frame[:, :, ::-1]

        if self.process_this_frame:
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            self.face_embeddings = face_recognition.face_encodings(rgb_small_frame, self.face_locations, model='small')
            
            self.face_names = []
            for face_embedding in self.face_embeddings:
                matches = face_recognition.compare_faces(self.embeddings, face_embedding)
                name = 'unknown'
                face_distances = face_recognition.face_distance(self.embeddings, face_embedding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.names[best_match_index]

                self.face_names.append(name)

        self.process_this_frame = not self.process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            # Draw a box around the face
            cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return img


WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

webrtc_streamer(
    key="face-recognition",
    video_transformer_factory=VideoTransformer,
    mode=WebRtcMode.SENDRECV,
    client_settings=WEBRTC_CLIENT_SETTINGS,
    async_transform = True
)
