## Задание по спецкурсу 'ИИ в индустриальных исследованиях'

Написать веб-сервис, который по базе знаний умеет отвечать на вопросы пользователя

### Установка
1. положить модель llama-2-7b-chat.Q4_K_M.gguf в директорию models
2. положить документы в директорию docs
3. запустить docker build -t container_name . для сборки контейнера

### Запуск
1. запустить docker run −p 8080:8080 container_name
2. дождаться загрузки эмбеддера с huggingface и инициализации LLaMA.
3. сервис доступен по эндопоинту POST /message с сообщением вида {"message": "user text" , "user_id": "1232"} 
