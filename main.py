#!/usr/bin/env python3

# Authors:  José Silva, Mário Vasconcelos, Nuno Cunha
# Nmec:     103268, 84081, 95167
# Email:    josesilva8@ua.pt, mario.vasconcelos@ua.pt, nunocunha99@ua.pt
# Version:  1.2
# Date:     27/10/2023

import cv2
import numpy as np
from os import listdir
import face_recognition
import copy 
from random import randint


from faceRecog import *
from track import *

# TODO: 
# .Deteção da saída de pessoas;
# .Melhorar performance;

#Notas: (ignorar)
#. Falta fazer o deepcopy;
#. Utilizar a classe detections;
#. fr = FaceRecognition()
#. Porquê fr.process_current_frame se a leitura de imagens é sequencial?
#. https://github.com/ageitgey/face_recognition
#. Separar Accuracy e JPG do nome
#. Retirar o .jpg do ficheiro


def w_text(image, text, pos):
    cv2.putText(image, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        
def main():
    # Parameters
    deactivate_threshold = 8.0 # secs
    delete_threshold = 2.0 # secs
    iou_threshold = 0.3
    tracking_padding = 0.75

    fr = FaceRecognition()
    video_frame_number = 0
    face_counter = 0
    tracks = []

    # Load web camera    
    cap = cv2.VideoCapture(0)
    
    # List all people on database
    all_known_people = []
    for people in os.listdir('faces'):
        all_known_people.append(people[:-4])

    while (cap.isOpened()):
        
        ret, image_rgb = cap.read()
        
        image_gui = copy.deepcopy(image_rgb)
        frame_stamp = round(float(cap.get(cv2.CAP_PROP_POS_MSEC))/1000,2)

        #Flip Frame
        image_gui = cv2.flip(image_gui,1)
        image_gray = cv2.cvtColor(image_gui, cv2.COLOR_BGR2GRAY)

        # Resize frame
        image_gui = cv2.resize(image_gui, (0,0), fx=0.5, fy=0.5)
        image_gray = cv2.resize(image_gray, (0,0), fx=0.5, fy=0.5)
        h, w, _ = image_gui.shape

        # Detect all frame faces
        fr.face_locations = face_recognition.face_locations(image_gui)
        fr.face_encodings = face_recognition.face_encodings(image_gui, fr.face_locations)
       
        # ------------------------------------------------------
        # Recognise and classify each face
        # ------------------------------------------------------
        fr.face_names = []
        fr.face_unknown = []
        for face_encoding in fr.face_encodings:
            matches = face_recognition.compare_faces(fr.known_face_encodings, face_encoding)
            
            # Face's Initial values
            name = 'Unknown' + str(face_counter)
            accuracy = 0
            unknown = True
 
            face_distances = face_recognition.face_distance(fr.known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                name = fr.known_face_names[best_match_index]
                accuracy = face_accuracy(face_distances[best_match_index])
                unknown = False

            fr.face_names.append(name)
            fr.face_accuracy.append(accuracy)
            fr.face_unknown.append(unknown)

        # ---------------------------------------------------------------------------------
        # Create list of current frame detections
        # ---------------------------------------------------------------------------------
        detections = []
        detection_idx = 0
        for (top, right, bottom, left), name, unknown in zip(fr.face_locations, fr.face_names, fr.face_unknown):      
            detection_id = 'D'+ str(video_frame_number) + '_' + str(detection_idx)
            detection_name = str(name)
            detection_unknown = unknown
            detection = Detection(left,right,top,bottom,detection_id,detection_name,detection_unknown,frame_stamp, image_gray)
            detections.append(detection)
            detection_idx += 1
        all_detections = copy.deepcopy(detections)

        # ------------------------------------------------------
        # Associate detections to existing tracks 
        # ------------------------------------------------------
        idxs_detections_to_remove = []
        active_detections = []
        for idx_detection, detection in enumerate(detections):
            active_detections.append(detection.detection_name)

            for track in tracks:  
                #If track is not active,do nothing;
                if not track.active:
                    continue
                
                # # Avoid attaching a known person to an unknown track   
                # if (detection.unknown == False) and (track.unknown == True):
                #     print('2nd option ('+str(track.track_name)+')')
                #     idxs_detections_to_remove.append(idx_detection)
                #     break

                # Attach detections and tracker with the same name
                if (detection.detection_name == track.track_name):
                    track.update(detection)
                    idxs_detections_to_remove.append(idx_detection)
                    break 
                    
                # Attach detection to current tracker using IOU
                iou = computeIOU(detection, track.detections[-1])
                if (iou > iou_threshold):
                    # If a detection is known and the tracker is unknown (happens when an unknown tracker is created first before the face is recognised)
                    if (detection.unknown == False) and (track.unknown == True): 
                        # Update the unknonw tracker with face data
                        track.track_name = detection.detection_name
                        track.known = True
                        idxs_detections_to_remove.append(idx_detection)
                        break
                    else:
                        # Just update the tracker
                        track.update(detection) # add detection to track
                        idxs_detections_to_remove.append(idx_detection)
                        break 

        # List non used detections to create new tracks
        idxs_detections_to_remove.reverse()
        for idx in idxs_detections_to_remove:
            del detections[idx]

        # --------------------------------------
        # Create new trackers with leftover detections
        # -------------------------------------
        for detection in detections: 
            color = (randint(0, 255), randint(0, 255), randint(0, 255))
            track = Track('T_'+str(face_counter), detection, color=color)
            tracks.append(track)
            face_counter += 1
            
            
        # --------------------------------------
        # Deactivate or eliminate tracks if last detection has been seen a long time ago
        # --------------------------------------
        idx_tracks_to_delete = []
        for idx_track, track in enumerate(tracks):
            time_since_last_detection = frame_stamp - track.detections[-1].stamp

            # Delete unknown tracks to avoid visual trash (common with bad detections)
            if (time_since_last_detection > delete_threshold) and (track.unknown == True):
                track.active = False
                idx_tracks_to_delete.append(idx_track)

            # De-activate known tracks after a long time
            if (time_since_last_detection > deactivate_threshold) and (track.unknown == False):                  
                track.active = False

        # Update track list only with active ones
        idx_tracks_to_delete.reverse()
        for idx in idx_tracks_to_delete:
            del tracks[idx]       

        # --------------------------------------
        # Template match existing trackers that did not have an associated detection in this frame
        # --------------------------------------

        # Generate a box to detect trackers leaving the frame
        # h, w, _ = image_gui.shape
        # tracking_padding = 0.25
        # track_rect_sp = (int(w*tracking_padding), int(h*tracking_padding))
        # track_rect_ep = (int(w - w*tracking_padding),int(h - h*tracking_padding))
        # cv2.rectangle(image_gui,track_rect_sp,track_rect_ep,(0,255,0),1)

        for track in tracks:
            if (track.detections[-1].stamp != frame_stamp): 
                track.track_template(image_gray, video_frame_number, frame_stamp)
    
        # ----------------------------------------------------------------------------------------
        # Visualization
        # ----------------------------------------------------------------------------------------
        # Draw detections
        for detection in all_detections:
            detection.draw(image_gui, (0,0,255))

        # Draw list of tracks
        for track in tracks:
            if (not track.active):
                continue
            track.draw(image_gui)
        
        # Add frame number and time to top left corner
        w_text(image_gui, 'Frame ' + str(video_frame_number), (10,20) )
        w_text(image_gui, 'Time ' + str(frame_stamp) + ' s',(10,45))
        w_text(image_gui, '_____________', (10,55))


        #Show active tracks
        active_tracks = []
        for idx, track in enumerate(tracks):
            if track.active:
                active_tracks.append(track.track_name)
        w_text(image_gui, 'Active tracks: ' +str(active_tracks),(10,h-10))      

        #Show active detections
        w_text(image_gui, 'Active Detections: ' + str(active_detections),(10,h-30))  
        
        # Show all known peolpe
        w_text(image_gui, 'Known People: ' + str(all_known_people),(10,h-50))  

        # Show missing people
        missing_people = []
        for people in all_known_people:
            if people in active_tracks:
                pass
            else:
                missing_people.append(people)
        w_text(image_gui, 'Missing People: ' + str(missing_people),(10,h-70)) 


        # View main frame
        cv2.imshow('Frame', image_gui)

        # Frame Commands
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break
        if k == ord('p'):
            cv2.waitKey(-1) 

        # Update frame number
        video_frame_number += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()