-- Migration script to update the admins table to match the new schema
-- This script will drop the existing admins table and create the new one

-- Drop existing admins table if it exists
DROP TABLE IF EXISTS `admins`;

-- Create the new Admins table with the provided schema
CREATE TABLE `Admins` (
    `admin_id` INT PRIMARY KEY AUTO_INCREMENT,
    `full_name` VARCHAR(100) NOT NULL,
    `email` VARCHAR(100) UNIQUE NOT NULL,
    `password_hash` VARCHAR(255) NOT NULL,
    `role` ENUM('Security', 'OSAS', 'Dean', 'Guidance') NOT NULL,
    `created_at` TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    `last_login` TIMESTAMP NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- Insert sample admin users (you can modify these as needed)
-- Default password for all sample users is 'admin123'
INSERT INTO `Admins` (`full_name`, `email`, `password_hash`, `role`) VALUES
('John Security', 'security@university.edu', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8QzK8K2', 'Security'),
('Jane OSAS', 'osas@university.edu', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8QzK8K2', 'OSAS'),
('Dr. Smith Dean', 'dean@university.edu', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8QzK8K2', 'Dean'),
('Ms. Guidance', 'guidance@university.edu', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj4J/8QzK8K2', 'Guidance');

-- Note: The password_hash values above are for 'admin123' 
-- You should change these passwords after first login
