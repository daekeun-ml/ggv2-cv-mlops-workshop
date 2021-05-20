# ggv2-cv-mlops-workshop

Caution: GGV2 Depolyment is not normally performed because the code is still under development.
## How to test
1. Please execute the Greengrass Setup of the reference url. https://greengrassv2.workshop.aws/en/chapter3_greengrasssetup.html
2. Copy all artifacts & recipes folders from this repository to the GreengrassCore folder.
3. Execute local inference with the command below. 

    ```
    cd /home/ubuntu/environment/GreengrassCore/artifacts/com.example.ImgClassification/1.0.0
    pip install -r requirements.txt
    python3 inference.py
    ```
