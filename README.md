# Time-Series Forecasting on Highway Traffic Volume and Speed (I)

## Introduction
This project is expected to be a long-term project, and this repository shows the achievement in Stage I. In Stage I, we succeeded in building a deep learning model that can estimate the traffic volume and speed on highways precisely in the short term.

## Methodology
We adopted the Convolutional Neural Network (CNN) architecture to include the spatial dependence and the time-series characteristic. We have not only collected the time-series traffic data of the target vehicle detector but also the data of the upstream and the downstream vehicle detector relative to the target vehicle detector.

In this project, we chose five features, respectively were:
- Time-Series __traffic volume__ data aggregated in summation in 5 minutes for the past 30 minutes
- Time-Series __traffic speed__ data aggregated on average in 5 minutes for the past 30 minutes
- Time-Series __traffic occupancy__ data aggregated on average in 5 minutes for the past 30 minutes
- __The number of lanes__ detected for each vehicle detectors
- __The characteristics of the road section__ for each vehicle detectors

These five features served as a channel respectively, then stacked them all as an input of the CNN. There was composed of a $3 \times 6$ tensor for each channel.

This figure shows the workflow of this project.
![img](images/work-flow.PNG)