FROM python:3.10

# Do not use gpu and cuda binaries
RUN pip install --no-cache-dir torch==2.1.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 8080

# Copy the content of the local src directory to the working directory
COPY . .

# Specify the command to run on container start
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]