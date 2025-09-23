# リポジトリ直下（pub_wisper/）
docker build -t pub-wisper .
docker run -p 8000:8000 pub-wisper

curl http://localhost:8000/                     # -> {"ok": true}
curl -X POST -F "file=@test.wav" -F "language=ja" -F "srt=1" \
  http://localhost:8000/api/transcribe -o out.srt

今後の改良案、
・フロントをNextでかく
・ GPTのAPIとか使って抽出した文章を整理させてその裏でその文章を別AIが検証するみたいなので言語化した文章の精度を高めるとかどうか
main.pyをエントリーポイントにして
asr.pyで音声処理
render.yamlでrenderデプロイ時に設定読み込ませられる
