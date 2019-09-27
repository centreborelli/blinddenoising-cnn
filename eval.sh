
sigma=20
root=fusion/
for seq in $root/*/take_*/; do
    args="--invis fusion$sigma --inir fusion --predict vis"
    out=${seq/fusion/results/n2n_vis_sigma$sigma/}
    rm -r $out; mkdir -p $out
    echo python main.py test $seq $out --model models/n2n_vis_sigma$sigma/checkpoint_59.tar $args

    out=${seq/fusion/results/n2n_visir_sigma$sigma/}
    rm -r $out; mkdir -p $out
    echo python main.py test $seq $out --model models/n2n_visir_sigma$sigma/checkpoint_59.tar $args

    args="--invis fusion --inir fusion$sigma --predict ir"
    out=${seq/fusion/results/n2n_ir_sigma$sigma/}
    rm -r $out; mkdir -p $out
    echo python main.py test $seq $out --model models/n2n_ir_sigma$sigma/checkpoint_59.tar $args

    out=${seq/fusion/results/n2n_irvis_sigma$sigma/}
    rm -r $out; mkdir -p $out
    echo python main.py test $seq $out --model models/n2n_irvis_sigma$sigma/checkpoint_59.tar $args
done | parallel -j 4

