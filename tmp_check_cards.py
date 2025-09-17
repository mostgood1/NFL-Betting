from app import app, _load_games
import re

def main():
    client = app.test_client()
    resp = client.get('/?season=2025&week=1')
    html = resp.get_data(as_text=True)
    card_count = len(re.findall(r'class="card"', html))
    team_block = html.count('team-block')
    games = _load_games()
    w1 = games[games.get('week') == 1]
    print('HTTP status', resp.status_code)
    print('Week1 games (loader)        :', len(w1))
    print('Rendered <div class="card">  :', card_count)
    print('team-block occurrences      :', team_block)
    if card_count:
        # quick sanity: extract first card snippet
        first = html.split('class="card"',1)[1][:600]
        print('First card snippet:\n', first)
    else:
        has_no_data_msg = 'No predictions found' in html
        print('No card divs found. Template no-data message present:', has_no_data_msg)

if __name__ == '__main__':
    main()
