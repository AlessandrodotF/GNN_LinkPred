# base imagePython
FROM python:3.11.11

# set the root folder 
WORKDIR /


# Iinstall requirements
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

#copy all
COPY ./ ./

# permission for entrypoint execution
RUN chmod +x ./entrypoint.sh

# ENTRYPOINT con exec form 
ENTRYPOINT ["./entrypoint.sh"]
