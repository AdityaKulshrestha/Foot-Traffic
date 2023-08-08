import csv
import time


def log_data(move_in, in_time, move_out, out_time):
    # function to log the counting data
    data = [move_in, in_time, move_out, out_time]
    # transpose the data to align the columns properly

    with open('utils/data/logs/counting_data.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        if myfile.tell() == 0:  # check if header rows are already existing
            wr.writerow(("Move In", "In Time", "Move Out", "Out Time"))
            wr.writerows(data)



def count_fps(prev_frame_time)
    new_frame_time = time.time()

    # Calculating the fps

    # fps will be number of frame processed in given time frame
    # since their will be most of time error of 0.001 second
    # we will be subtracting it to get more accurate result
    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time

    # converting the fps into integer
    fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    fps = str(fps)

