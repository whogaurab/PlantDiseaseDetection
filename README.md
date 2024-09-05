# PlantDiseaseDetection

this is a plant diseasedetection end to end MLOPS project built using YOLOv9. first of all i collected the data from multiple internet sources of rice plant leaf having following categories 'Bacterial Leaf Blight', 'Brown Spot', 'Healthy', 'Leaf Blast', 'Leaf Blight', 'Leaf Scald', 'Leaf Smut', 'Narrow Brown Spot'.we fine tuned the model using goggle collab the plant disease detection ipynb file has been saved in research directory.now we come to our local machine and follow the following instructon as i have written below. 
create an virtual enviromrnt first

    conda create -p venv python==3.11 -y
    conda activate venv/

install the requirements using the following command:

    pip install -r requirements.txt

insert your best.pt yolov9 model into yolov9 folder and run the following command

    python app.py

you can see youur app is running on localhost:8080