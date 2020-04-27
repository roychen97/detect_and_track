## Environment

#### install tensorflow 1.15
change layer setting for TX2
[layer setting](https://ulsee01-my.sharepoint.com/:w:/g/personal/suchin_chiu_ulsee_com/EfDAj5LeNVxBg5WyBiILU3IBtpvwn9H4tN-JRYfZjJiGnA?e=sbImHQ)

or

#### use docker
1.edit docker/Makefile


  edit MOUNT_DIR path to workspace and datasets

2.build & run


    cd docker
    sudo make build
    sudo make x11
    
  
##  Steps

    python3 dataset/convert_allperson_tfrecords.py
    python3 train_ssd.py
    python3 simple_ssd_demo.py
   
 details in [online document](https://ulsee01-my.sharepoint.com/:w:/g/personal/suchin_chiu_ulsee_com/EbCPBw4vvN9KmB1vRGJuNJYBUBMIc_1gjq5ikCRiT34w0A?e=07Gdaz)