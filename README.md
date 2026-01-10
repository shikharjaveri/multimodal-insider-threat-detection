# Multimodal Insider Threat Detection System

This project trains an LSTM-based insider threat detection model using daily user activity features from the CERT dataset. For each user, 7-day sequences of logins, file activity, USB usage, web uploads, and email behavior are used to predict whether that user will perform suspicious activity in the near future. The model learns how behavior changes over time and assigns risk based on patterns that appear in the days leading up to a potential data exfiltration event.
