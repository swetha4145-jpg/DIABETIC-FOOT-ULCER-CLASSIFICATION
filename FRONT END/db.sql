DROP DATABASE IF EXISTS Ulcer;
CREATE DATABASE Ulcer;
USE Ulcer;

CREATE TABLE users (
    id INT PRIMARY KEY AUTO_INCREMENT,
    name VARCHAR(225),
    email VARCHAR(225),
    password VARCHAR(225)
);

CREATE TABLE predictions (
    id INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255),
    image_path VARCHAR(255),
    prediction VARCHAR(50),
    confidence FLOAT,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);