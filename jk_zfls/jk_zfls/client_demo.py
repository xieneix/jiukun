import socketio
import time
import numpy as np
import os
import matplotlib.pyplot as plt
from src.cut import cut_image
from src.open_max import classify, load_model
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm  # 可选：用于显示进度条

def get_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_shape(lst[0]) if lst else []
    return []

def action_policy(action_shape):
    # 0: down, loc+=[1,0]
    # 1: right, loc+=[0,1]
    # 2: up, loc+=[-1,0]
    # 3: left, loc+=[0,-1]
    # 4: collect
    return np.random.randint(action_shape)


def recognition(img, model_path='model/resnet18_finetuned.pth', preview=False):
    """
    将整张图片分块并对每个块进行分类。
    """
    # 加载模型和特征提取器
    _, feature_extractor = load_model(model_path)

    # 切割图片
    img = np.array(img)
    patches = cut_image(img, preview=preview)

    # 初始化 12x12 的空数组
    result_array = np.zeros((12, 12), dtype=int)

    def classify_patch(args):
        idx, patch = args
        row = idx // 12
        col = idx % 12
        # 调用 classify 函数进行分类
        result = classify(patch, idx, feature_extractor)
        return row, col, result

    # 使用线程池并行处理
    with ThreadPoolExecutor(max_workers=15) as executor:
        tasks = [(idx, patch) for idx, patch in enumerate(patches)]
        for row, col, result in tqdm(executor.map(classify_patch, tasks), total=len(tasks)):
            result_array[row, col] = result

    return result_array


def team_play_game(team_id, game_type, game_data_id, ip, port):
    sio = socketio.Client(request_timeout=60)
    grid = None
    begin = game_type + game_data_id
    @sio.event
    def connect():
        #print(f"Connected to server, game_type: {game_type}, game data id: {begin}")
        pass
    @sio.event
    def disconnect():
        #print(f"End game {begin}, disconnected from server")
        pass
    @sio.event
    def connect_error(data):
        print('Connect error', data)
    @sio.event
    def disconnect_error(data):
        print('Disconnect error', data)
    @sio.event
    def response(data):
        nonlocal grid
        if 'error' in data:
            print(data['error'])
            sio.disconnect()
        else:
            try:
                if data['rounds'] == 0:
                    print(f"Team {data['team_id']} begin game {data['game_id']}")
                is_end = data.get('is_end', False)
                score = data['score']
                bag = data['bag']
                loc = data['loc']
                game_id = data['game_id']
                os.makedirs(f'./{data["team_id"]}/', exist_ok=True)
                send_data = {'team_id': data['team_id'], 'game_id': game_id}
                if data['rounds']==0:
                    if (game_type == 'a'):
                        grid = np.array(data['grid'], dtype=int)
                        print(grid.shape)
                    if (game_type == '2'):

                        grid = recognition(data['img'])
                        print(grid)
                        send_data['grid_pred'] = grid.tolist()
                score_npy = f'./{data["team_id"]}/{data["game_id"]}_score.npy'
                if os.path.exists(score_npy):
                    prev_score = np.load(score_npy)
                else:
                    prev_score = np.array(0.0)
                np.save(score_npy, prev_score + score)
                if is_end:
                    print(f"Team {data['team_id']} end game {data['game_id']}, cum_score: {prev_score + score:.2f}")
                    if game_type == '2':
                        print(f'Recognition acc on this game fig: {data["acc"]}')
                    sio.disconnect()
                else:
                    action = action_policy(5)
                    if action == 4:
                        grid[loc[0], loc[1]] = -1
                    send_data['action'] = action
                    if sio.connected:
                        sio.emit('continue', send_data)
                    else:
                       print('sio not connected')
            except Exception as e:
                print(f'{e}')
                sio.disconnect()
    try:
        # 连接到服务器
        sio.connect(f'http://{ip}:{port}/', wait_timeout=30)
        # 发送消息到服务器
        message = {'team_id': team_id, 'begin': begin}
        sio.emit('begin', message)
        sio.wait()
    except socketio.exceptions.ConnectionError as e:
        print('Connection Error')
        sio.disconnect()
    except Exception as e:
        print(f'Exception: {e}')
        sio.disconnect()
    finally:
        #print('end team play game')
        pass


if __name__ == '__main__':
    team_id = 'libms897ww51'
    ip = '52.82.16.74'
    port = '8090'
    # game_type must be in ['2', 'a'], '2' for full game and recognition only, 'a' for action_only
    game_type = '2'
    
    # 初赛的第1阶段，game_data_id  must be in ['00000', '00001', ..., '00099']
    # 初赛的终榜阶段，game_data_id  must be in ['00000', '00001', ..., '00199']
    game_data_id = [f'{i:05}' for i in range(1, 11)]
    st = time.time()
    for gdi in game_data_id:
        team_play_game(team_id, game_type, gdi, ip, port)
    print(f'Total time: {(time.time()-st):.1f}s')
# 刘姥姥进大观园了
# 死老鼠来偷油
# 月半猫你有意见？
# 保佑意见
# 我补课
# 有问题问我
# ok
# 我先坐下ppt就来写
