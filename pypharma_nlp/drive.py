import sys


def download_drive_file(file_id, destination_path):
    
    """Download a file from google drive."""
    
    if "google.colab" in sys.modules:
        _download_drive_file_colab(file_id, destination_path)
    else:
        _download_drive_file_manual(file_id, destination_path)


def _download_drive_file_colab(file_id, destination_path):
    
    # Authenticate and create the PyDrive client.
    from google.colab import auth
    from oauth2client.client import GoogleCredentials
    from pydrive.auth import GoogleAuth
    from pydrive.drive import GoogleDrive
    auth.authenticate_user()
    gauth = GoogleAuth()
    gauth.credentials = GoogleCredentials.get_application_default()
    drive = GoogleDrive(gauth)
    
    # Download TAR
    print ("Downloading tar file")
    myzip = drive.CreateFile({ "id" : file_id })
    myzip.GetContentFile(destination_path)


def _download_drive_file_manual(file_id, destination_path):
    
    # Create directory
    print("Please download the '%s' file from this URL:\nhttps://drive.google.com/file/d/%s/view?usp=sharing" % (destination_path, file_id))
    print("Place the file in '%s'." % destination_path)
    print("Press Enter when done.")
    input()
