
mkdir -p camouflage; cd camouflage
for i in 1 2 3 4 5; do
wget http://www02.smt.ufrj.br/~fusion/Database/Registered/Camouflage/take_$i.tar.gz &
done
cd ..

mkdir -p lab; cd lab
for i in 1 2 3 4 5; do
wget http://www02.smt.ufrj.br/~fusion/Database/Registered/Lab/take_$i.tar.gz &
done
cd ..

mkdir -p patio; cd patio
wget http://www02.smt.ufrj.br/~fusion/Database/Registered/Patio/take_1.tar.gz &
cd ..

mkdir -p guanabara; cd guanabara
for i in 1 2 3; do
wget http://www02.smt.ufrj.br/~fusion/Database/Registered/Guanabara_Bay/take_$i.tar.gz &
done
cd ..

mkdir -p hangar; cd hangar
for i in 1 2 3 4; do
wget http://www02.smt.ufrj.br/~fusion/Database/Registered/Hangar/take_$i.tar.gz &
done
cd ..


mkdir -p trees; cd trees
for i in 1 2 3 4; do
wget http://www02.smt.ufrj.br/~fusion/Database/Registered/Trees/take_$i.tar.gz &
done
cd ..

wait

