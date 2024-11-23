import requests
import os
import sys


url = 'http://69.230.243.237:8081/'


def test_send_recv(url, data):
    # send action, receive new game data
    try:
        response = requests.post(url, json=data)
        print(f'Response: {response.text}, Status code: {response.status_code}')
    except Exception as e:
        print(f'Error occurred: {e}')


def team_test_connection(team_id, url):
    print('='*40 + f'\nTeam id: {team_id} test!!!')
    test_send_recv(url, {'team_id': team_id,})


if __name__=="__main__":
    team_id = 'libms897ww51'
    team_test_connection(team_id, url)
