#pragma once
/* #region header */
#pragma GCC optimize("Ofast")
#include <bits/stdc++.h>
using namespace std;
// types
using ll = long long;
using ull = unsigned long long;
using ld = long double;
typedef pair<ll, ll> Pl;
typedef pair<int, int> Pi;
typedef vector<ll> vl;
typedef vector<int> vi;
typedef vector<char> vc;
template <typename T>
using mat = vector<vector<T>>;
typedef vector<vector<int>> vvi;
typedef vector<vector<long long>> vvl;
typedef vector<vector<char>> vvc;
// abreviations
#define all(x) (x).begin(), (x).end()
#define rall(x) (x).rbegin(), (x).rend()
#define rep_(i, a_, b_, a, b, ...) for (ll i = (a), max_i = (b); i < max_i; i++)
#define rep(i, ...) rep_(i, __VA_ARGS__, __VA_ARGS__, 0, __VA_ARGS__)
#define rrep_(i, a_, b_, a, b, ...) \
    for (ll i = (b - 1), min_i = (a); i >= min_i; i--)
#define rrep(i, ...) rrep_(i, __VA_ARGS__, __VA_ARGS__, 0, __VA_ARGS__)
#define srep(i, a, b, c) for (ll i = (a), max_i = (b); i < max_i; i += c)
#define SZ(x) ((int)(x).size())
#define pb(x) push_back(x)
#define eb(x) emplace_back(x)
#define mp make_pair
//入出力
#define print(x) cout << x << endl
template <class T>
ostream &operator<<(ostream &os, const vector<T> &v)
{
    for (auto &e : v)
        cout << e << " ";
    cout << endl;
    return os;
}
void scan(int &a) { cin >> a; }
void scan(long long &a) { cin >> a; }
void scan(char &a) { cin >> a; }
void scan(double &a) { cin >> a; }
void scan(string &a) { cin >> a; }
template <class T>
void scan(vector<T> &a)
{
    for (auto &i : a)
        scan(i);
}
#define vsum(x) accumulate(all(x), 0LL)
#define vmax(a) *max_element(all(a))
#define vmin(a) *min_element(all(a))
#define lb(c, x) distance((c).begin(), lower_bound(all(c), (x)))
#define ub(c, x) distance((c).begin(), upper_bound(all(c), (x)))
// functions
// gcd(0, x) fails.
ll gcd(ll a, ll b) { return b ? gcd(b, a % b) : a; }
ll lcm(ll a, ll b) { return a / gcd(a, b) * b; }
template <class T>
bool chmax(T &a, const T &b)
{
    if (a < b)
    {
        a = b;
        return 1;
    }
    return 0;
}
template <class T>
bool chmin(T &a, const T &b)
{
    if (b < a)
    {
        a = b;
        return 1;
    }
    return 0;
}
template <typename T>
T mypow(T x, ll n)
{
    T ret = 1;
    while (n > 0)
    {
        if (n & 1)
            (ret *= x);
        (x *= x);
        n >>= 1;
    }
    return ret;
}
ll modpow(ll x, ll n, const ll mod)
{
    ll ret = 1;
    while (n > 0)
    {
        if (n & 1)
            (ret *= x);
        (x *= x);
        n >>= 1;
        x %= mod;
        ret %= mod;
    }
    return ret;
}
ll safemod(ll x, ll mod) { return (x % mod + mod) % mod; }
int popcnt(ull x) { return __builtin_popcountll(x); }
template <typename T>
vector<int> IOTA(vector<T> a)
{
    int n = a.size();
    vector<int> id(n);
    iota(all(id), 0);
    sort(all(id), [&](int i, int j)
         { return a[i] < a[j]; });
    return id;
}
long long xor64(long long range) {
    static uint64_t x = 88172645463325252ULL;
    x ^= x << 13;
    x ^= x >> 7;
    return (x ^= x << 17) % range;
}
struct Timer
{
    clock_t start_time;
    void start() { start_time = clock(); }
    int lap()
    {
        // return x ms.
        return (clock() - start_time) * 1000 / CLOCKS_PER_SEC;
    }
};
template <typename T = int>
struct Edge
{
    int from, to;
    T cost;
    int idx;

    Edge() = default;

    Edge(int from, int to, T cost = 1, int idx = -1)
        : from(from), to(to), cost(cost), idx(idx) {}

    operator int() const { return to; }
};

template <typename T = int>
struct Graph
{
    vector<vector<Edge<T>>> g;
    int es;

    Graph() = default;

    explicit Graph(int n) : g(n), es(0) {}

    size_t size() const { return g.size(); }

    void add_directed_edge(int from, int to, T cost = 1)
    {
        g[from].emplace_back(from, to, cost, es++);
    }

    void add_edge(int from, int to, T cost = 1)
    {
        g[from].emplace_back(from, to, cost, es);
        g[to].emplace_back(to, from, cost, es++);
    }

    void read(int M, int padding = -1, bool weighted = false,
              bool directed = false)
    {
        for (int i = 0; i < M; i++)
        {
            int a, b;
            cin >> a >> b;
            a += padding;
            b += padding;
            T c = T(1);
            if (weighted)
                cin >> c;
            if (directed)
                add_directed_edge(a, b, c);
            else
                add_edge(a, b, c);
        }
    }
};

/* #endregion*/
// constant
#define inf 1000000000ll
#define INF 4000000004000000000LL
#define endl '\n'
const long double eps = 0.000000000000001;
const long double PI = 3.141592653589793;


/**
 * @brief UnionFind
 * @docs docs/UnionFind.md
 */
struct UnionFind {
    vector<int> data;  // sizes of sets

    UnionFind(int sz) : data(sz, -1) {}

    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (data[x] > data[y]) swap(x, y);
        data[x] += data[y];
        data[y] = x;
        return true;
    }

    int find(int k) {
        if (data[k] < 0) return k;
        return data[k] = find(data[k]);
    }

    int size(int k) { return (-data[find(k)]); }

    bool same(int x, int y) { return find(x) == find(y); }
};

/**
 * @brief start_temp: 一回の遷移で動きうるスコア幅の最大値程度, end_temp: 一回の遷移で動きうるスコア幅の最小値程度
 */

template<typename State, typename Action>
State simulated_annealing(State state, double start_temp, double end_temp, double time_limit, bool minimize=false) {
    Timer timer;
    timer.start();
    rep(t, 1e9) {
        double time = timer.lap();
        double temp = start_temp + (end_temp - start_temp) * time / time_limit;
        if(time > time_limit) break;

        double score = state.score;
        Action action = state.generate_action();
        state.step(action);
        double new_score = state.score;
        double diff = new_score - score;
        if(minimize) diff *= -1;
        double prob = exp(diff / temp);
        if (prob < (double) xor64(10000000) / 10000000) {
            state.rollback();
        }
        if(t % 100 == 0) cerr << t << ": " << state.score << endl;
    }
    return state;
}

template<typename State, typename Action>
State hill_climbing(State state, double time_limit, bool minimize=false){
    Timer timer;
    timer.start();
    rep(t, 1e9) {
        double time = timer.lap();
        if(time > time_limit) break;

        double score = state.score;
        Action action = state.generate_action();
        state.step(action);
        double new_score = state.score;
        double diff = new_score - score;
        if(minimize) diff *= -1;
        if (diff < 0) {
            state.rollback();
        }
        if(t % 100 == 0) cerr << t << ": " << state.score << endl;
    }
    return state;
}

const ll max_P = 5000;
ll N, M, K;
vl x, y, u, v, w, a, b;
mat<ll> dist, sorted_by_dist;

ll ll_sqrt(ll val){
    ll ok = 1e5, ng = 0;
    while(ok - ng > 1){
        ll mid = (ok + ng) / 2;
        if(mid * mid >= val){
            ok = mid;
        }else{
            ng = mid;
        }
    }
    return ok;
}

ll calc_dist(ll i, ll j){
    ll dist2 = (a[i] - x[j]) * (a[i] - x[j]) + (b[i] - y[j]) * (b[i] - y[j]);
    return ll_sqrt(dist2);
}

void input(){
    cin >> N >> M >> K;
    x.resize(N), y.resize(N);
    u.resize(M), v.resize(M), w.resize(M);
    a.resize(K), b.resize(K);
    rep(i, N) cin >> x[i] >> y[i];
    rep(i, M) {
        cin >> u[i] >> v[i] >> w[i];
        u[i]--, v[i]--;
    }
    rep(i, K) cin >> a[i] >> b[i];
    dist.resize(K, vl(N));
    rep(i, K)rep(j, N){
        dist[i][j] = calc_dist(i, j);
    }
    sorted_by_dist.resize(N);
    rep(i, N){
        vl dist_from_i(K);
        rep(j, K){
            dist_from_i[j] = dist[j][i];
        }
        for(ll j: IOTA(dist_from_i)){
            sorted_by_dist[i].pb(j);
        }
    }
}

ll calc_cost(vl& P, vl& B){
    ll S = 0;
    rep(i, N){
        S += P[i] * P[i];
    }
    rep(i, M){
        S += B[i] * w[i];
    }
    return S;
}

ll calc_score(vl& P, vl& B){
    ll S = calc_cost(P, B);
    return 1e6 * (1 + 1e8 / (S + 1e7));
}

void output(vl P, vl B){
    rep(i, N) cout << P[i] << ' ';
    cout << endl;
    rep(i, M) cout << B[i] << ' ';
    cout << endl;
    cerr << "Score = " << calc_score(P, B) << endl;
    cerr << "Cost = " << calc_cost(P, B) << endl;
}

pair<bool, vl> spanning_tree(vl V){
    vi id = IOTA(w);
    UnionFind uf(N);
    vl B(M, 0);
    for(int i: id){
        if(V[u[i]] == 1 && V[v[i]] == 1 && uf.unite(u[i], v[i])){
            B[i] = 1;
        }
    }

    vl stations;
    bool connected = true;
    rep(i, N){
        if(V[i] == 1) stations.pb(i);
    }
    for(int id: stations){
        if(!uf.same(0, id)) connected = false;
    }
    return {connected, B};
}

pair<mat<ll>, vl> assign_greedy(vl V){
    vl P(N);
    mat<ll> assignment(N);
    rep(i, K){
        ll min_cost = inf, min_station = -1;
        rep(j, N){
            ll d = dist[i][j];
            ll cost = max(0ll, d * d - P[j] * P[j]); 
            if(d <= max_P && V[j] == 1 && chmin(min_cost, cost)) min_station = j;
        }
        if(min_station != -1){
            chmax(P[min_station], dist[i][min_station]);
            assignment[min_station].pb(i);
        }
    }
    return {assignment, P};
}

struct Action{
    ll id;
};

struct State{
    bool valid;
    ll score;
    ll score_rollback;
    vl V, P, B;
    vl V_rollback, P_rollback, B_rollback;
    mat<ll> assignment;
    mat<ll> assignment_rollback;

    State(): V(N, 1){
        valid = true;
        bool connected;
        tie(connected, B) = spanning_tree(V);
        tie(assignment, P) = assign_greedy(V);
        score = calc_score(P, B);
    }

    Action generate_action(){
        return {xor64(N - 1) + 1};
    }

    void update_assignment(Action action){
        if(V[action.id] == 0){
            for(ll i: assignment[action.id]){
                ll min_cost = inf, min_station = -1;
                rep(j, N){
                    ll d = dist[i][j];
                    ll cost = max(0ll, d * d - P[j] * P[j]); 
                    if(d <= max_P && V[j] == 1 && chmin(min_cost, cost)) min_station = j;
                }
                if(min_station == -1){
                    valid = false;
                    return;
                }else{
                    assignment[min_station].pb(i);
                    chmax(P[min_station], dist[i][min_station]);    
                }
            }
            assignment[action.id].clear();
            P[action.id] = 0;
        }else{
            rep(j, N){
                vl tmp;
                for(ll i: assignment[j]){
                    ll d = dist[i][j];
                    ll new_d = dist[i][action.id];
                    if(new_d < d && P[j] == d){
                        assignment[action.id].pb(i);
                        chmax(P[action.id], new_d);
                    }else{
                        tmp.pb(i);
                    }
                }
                P[j] = 0;
                assignment[j] = tmp;
                for(ll i: assignment[j]) chmax(P[j], dist[i][j]);
            }
        }
    }

    void step(Action action){
        V_rollback = V, P_rollback = P, B_rollback = B;
        score_rollback = score;
        assignment_rollback = assignment;

        V[action.id] ^= 1;
        bool connected;
        tie(connected, B) = spanning_tree(V);
        if(!connected) valid = false;
        if(valid){
            update_assignment(action);
            score = calc_score(P, B);
        }else{
            score = -inf;
        }
    } 

    void rollback(){
        valid = true;
        V = V_rollback, P = P_rollback, B = B_rollback;
        score = score_rollback;
        assignment = assignment_rollback;
    }
};

struct Action_P{
    ll id, diff;
};

struct State_P{
    ll max_diff = 100;
    ll score = 0;
    vl V, B, P, cnt, max_covered;

    // for rollback
    ll id, val, pre_score;
    vl B_rollback;

    State_P(){
        V.resize(N);
        rep(i, N)V[i] = 1;
        mat<ll> assignment;
        tie(assignment, P) = assign_greedy(V);
        cnt.resize(K);
        max_covered.assign(N, -1);
        rep(i, K) rep(j, N){
            ll r_id = sorted_by_dist[j][i];
            if(dist[r_id][j] <= P[j]){
                cnt[r_id]++;
                chmax(max_covered[j], i);
            }
        }
        bool connected;
        tie(connected, B) = spanning_tree(V);
        rep(i, N){
            score += P[i] * P[i];
        }
        rep(i, M){
            score += B[i] * w[i];
        }
    }
    Action_P generate_action(){
        return {xor64(N), xor64(max_diff * 2) - max_diff};
    }
    bool is_covered2(ll p, ll p_pre){
        rep(i, K){
            bool covered = false;
            rep(j, N){
                if(dist[i][j] <= P[j]) covered = true;
            }
            if(!covered) return false;
        }
        return true;
    }   

    bool is_covered(ll p, ll p_pre){
        bool covered = true;
        if(p >= p_pre){
            rep(i, max_covered[id] + 1, K){
                int r_id = sorted_by_dist[id][i];
                if(dist[r_id][id] > p){
                    break;
                }
                cnt[r_id]++;
                max_covered[id] = i;        
            }
        }else{
            ll tmp = max_covered[id];
            max_covered[id] = -1;
            rrep(i, 0, tmp + 1){
                int r_id = sorted_by_dist[id][i];
                if(dist[r_id][id] <= p){
                    max_covered[id] = i;        
                    break;
                }
                cnt[r_id]--;
                if(cnt[r_id] == 0) covered = false;
            }
        }
        return covered;
    }   

    void step(Action_P action){
        // 保存
        id = action.id;
        val = P[id];
        pre_score = score;

        // 状態を変更
        P[id] += action.diff;
        chmax(P[id], 0ll), chmin(P[id], max_P);
        if(P[id] == 0) V[id] = 0;
        else V[id] = 1;
        bool covered = is_covered(P[id], val);
        if(((val > 0 && P[id] == 0) || (val == 0 && P[id] > 0))){
            bool connected;
            swap(B, B_rollback);
            tie(connected, B) = spanning_tree(V);
            if(connected && covered){
                score = calc_cost(P, B);
            }else{
                score = INF;
            }
        }else{
            // スコアの更新
            if(covered){
                score += P[id] * P[id] - val * val;
            }else{
                score = INF;
            }
        }
    }
    void rollback(){
        if(val == 0) V[id] = 0;
        else V[id] = 1;
        bool covered = is_covered(val, P[id]);
        if(((val > 0 && P[id] == 0) || (val == 0 && P[id] > 0))){
            swap(B, B_rollback);
        }
        P[id] = val;
        score = pre_score;
    }
};

void solve_use_all(){
    vl V(N, 1);
    auto[assignment, P] = assign_greedy(V);
    auto [connected, B] = spanning_tree(V);
    cerr << connected << endl;
    output(P, B);
}

void solve_hill_climbing(){
    State state;
    // state = hill_climbing<State, Action>(state, 1800);
    state = simulated_annealing<State, Action>(state, 1000, 100, 1800);
    auto[assignment, P] = assign_greedy(state.V);
    auto [connected, B] = spanning_tree(state.V);
    output(state.P, B);
}

void solve_hill_climbing_P(){
    State_P state;
    // state = hill_climbing<State_P, Action_P>(state, 1800, true);
    state = simulated_annealing<State_P, Action_P>(state, 1000000, 1000, 1800, true);
    output(state.P, state.B);
}

int main(int argc, char *argv[]) {
    cin.tie(0);
    ios::sync_with_stdio(0);
    cout << setprecision(30) << fixed;
    cerr << setprecision(30) << fixed;
    input();
    // solve_use_all();
    // solve_hill_climbing();
    solve_hill_climbing_P();
}