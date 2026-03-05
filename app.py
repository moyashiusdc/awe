#플라스크 jwt 페이로드(datetime)
from flask import Flask, render_template, jsonify, request
from pymongo import MongoClient
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
import datetime

app = Flask(__name__)

SECRET_KEY = 'secret_key'

#일단 아무주소나
client = MongoClient('mongodb+srv://sparta_db_user:[EMAIL_ADDRESS]/?appName=Cluster0')

# 13번 라인 뒤에 이 두 줄만 남기고 나머지 한글 설명은 지우세요
db = client.dbsparta 

@app.route('/')
def home():
    return render_template('index.html')

return jsonify({'result': 'success', 'token': token})

@app.route('/')
def home():
    return render_template('index.html')

#해시 + 솔트는 그냥넘어감
@app.route('/api/register', methods=['POST'])
def register():
    user_id = request.form['id_give']
    user_pw = request.form['pw_give']

    pw_hash = generate_password_hash(user_pw)
    db.users.insert_one({'id': user_id, 'pw': pw_hash})
    return jsonify({'msg': '회원가입완료'})

@app.route('/api/login', methods=['POST'])
def login():
    user_id = request.form['id_give']
    user_pw = request.form['pw_give']
    return jsonify({'msg': '로그인완료'})

@app.route('/api/login', methods=['POST'])
def login():
    user_id = request.form['id_give']
    user_pw = request.form['pw_give']

    # 1. DB에서 유저 찾기
    user = db.users.find_one({'id': user_id})

    # 2. 비번 체크 및 토큰 발행
    if user and check_password_hash(user['pw'], user_pw):
        payload = {
            'id': user_id,
            'exp': datetime.datetime.utcnow() + datetime.timedelta(minutes=30)
        }
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')

        # 자바스크립트가 기다리는 'result': 'success'를 꼭 넣어줘야 함!
        return jsonify({'result': 'success', 'token': token})
    else:
        return jsonify({'result': 'fail', 'msg': '아이디/비번을 확인해주세요.'})

if __name__ == '__main__':
    app.run(debug=True)