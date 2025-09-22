# リポジトリ直下（pub_wisper/）
docker build -t pub-wisper .
docker run -p 8000:8000 pub-wisper

curl http://localhost:8000/                     # -> {"ok": true}
curl -X POST -F "file=@test.wav" -F "language=ja" -F "srt=1" \
  http://localhost:8000/api/transcribe -o out.srt
