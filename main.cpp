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

ll N, M, K;
vl x, y, u, v, w, a, b;

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
}

ll calc_score(vl P, vl B){
    ll S = 0;
    rep(i, N){
        S += P[i] * P[i];
    }
    rep(i, M){
        S += B[i] * w[i];
    }
    return 1e6 * (1 + 1e8 / (S + 1e7));
}

void output(vl P, vl B){
    rep(i, N) cout << P[i] << ' ';
    cout << endl;
    rep(i, M) cout << B[i] << ' ';
    cout << endl;
    cerr << "Score = " << calc_score(P, B) << endl;
}

vl spanning_tree(vl V){
    vi id = IOTA(w);
    UnionFind uf(N);
    vl B(M, 0);
    for(int i: id){
        if(V[u[i]] == 1 && V[v[i]] == 1 && uf.unite(u[i], v[i])){
            B[i] = 1;
        }
    }
    return B;
}

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

ll dist(ll i, ll j){
    ll dist2 = (a[i] - x[j]) * (a[i] - x[j]) + (b[i] - y[j]) * (b[i] - y[j]);
    return ll_sqrt(dist2);
}

vl assign_greedy(vl V){
    vl P(N);
    rep(i, K){
        ll min_dist = inf, min_station = -1;
        rep(j, N){
            if(chmin(min_dist, dist(i, j))) min_station = j;
        }
        chmax(P[min_station], min_dist);
    }
    return P;
}

void solve_use_all(){
    vl V(N, 1);
    vl P = assign_greedy(V);
    vl B = spanning_tree(V);
    output(P, B);
}


int main(int argc, char *argv[]) {
    cin.tie(0);
    ios::sync_with_stdio(0);
    cout << setprecision(30) << fixed;
    cerr << setprecision(30) << fixed;
    input();
    solve_use_all();
}