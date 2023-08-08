# Foot Traffic Counter

The Foot Traffic Counter is a simple project that counts the number of people entering a location, such as a mall, and keeps track of the data. It operates over a socket connection, allowing seamless integration with various systems. This README provides a user-friendly guide to help you install and set up the project.

Please note that the videos used for testing and demonstration purposes are not owned by the project owner. Make sure you have the necessary rights to use any videos you provide for testing.

## Installation

1. Clone the repository to your local machine:
   ```
   git clone https://github.com/AdityaKulshrestha/Foot-Traffic.git
   cd foot-traffic-counter
   ```
   
2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```commandline
   pip install -r requirements.txt
   ```
4. Run the project 
   ```commandline
   python main.py
   ```
   
## Usage
1. Start the Foot Traffic Counter by running main.py.

2. The program will establish a socket connection to receive video frames from a source (camera or video file) and count the number of people entering the location.

3. To stop the program, press the 'q' key. The program will terminate, and you will see the total number of people counted during the session.

## Disclaimer
The videos used for testing and demonstration are not owned by the project owner. 

## Contributing
Contributions are welcome! If you find a bug or have suggestions for improvements, please open an issue or submit a pull request.

## License
This project is licensed under the MIT License.

Happy counting and tracking foot traffic! 🚶‍♂️🚶‍♀️
