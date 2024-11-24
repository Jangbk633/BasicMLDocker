clone 이후 해당 directory에서

docker build -t image_name .

실행 후 완료되면

docker run --name container_name -p 8000:8000 image_name

실행 후 완료되면 크롬 등의 웹을 켜서

http://localhost:8000/

로 접속 후 텍스트 입력하면 됩니다
