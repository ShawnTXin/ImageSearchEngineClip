FROM python:3.9-slim
ADD . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["CLIP_streamlit.py"]
