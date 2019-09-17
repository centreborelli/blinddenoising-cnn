
sigma=20

# (noisy vis, ir) -> noisy vis
args="--invis fusion$sigma --inir fusion --outvis fusion$sigma --outir fusion"
out=models/n2n_visir_sigma$sigma/
rm -r $out; mkdir -p $out
python3 main.py train . --inchannels 2 --outchannels 1 --lr 1e-3 --saveto $out $args

# noisy vis -> noisy vis
out=models/n2n_vis_sigma$sigma/
rm -r $out; mkdir -p $out
python3 main.py train . --inchannels 1 --outchannels 1 --lr 1e-3 --saveto $out $args

# noisy ir -> noisy ir
args="--invis fusion --inir fusion$sigma --outvis fusion --outir fusion$sigma"
out=models/n2n_ir_sigma$sigma/
rm -r $out; mkdir -p $out
python3 main.py train . --inchannels 1 --outchannels 1 --lr 1e-3 --saveto $out --predict ir $args

# (noisy ir, vis) -> noisy ir
out=models/n2n_irvis_sigma$sigma/
rm -r $out; mkdir -p $out
python3 main.py train . --inchannels 2 --outchannels 1 --lr 1e-3 --saveto $out --predict ir $args

