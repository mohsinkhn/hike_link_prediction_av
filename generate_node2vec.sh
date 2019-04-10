python prepare_data_node2vec.py
git clone https://github.com/snap-stanford/snap.git
cd snap
make all
cd examples/node2vec
./node2vec -i:../../../utility/weighted_nodes.graph -o:../../../utility/hike_node2vec.emb -v -l:21 -d:32 -p:0.3 -e:2 -k:4
cd ../../..
