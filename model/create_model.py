from model.SpaNichePair import SpaNichePair


def create_model(model_name, args):
    model_name = str(model_name).lower()
    latent_dim = int(args.n_z)

    if model_name != "spanichepair_gcnii":
        raise ValueError(
            f"Unsupported model_name: {model_name}. This release exposes the complete spaNicheGraph model only."
        )

    return SpaNichePair(
        ae_n_enc_1=128,
        ae_n_enc_2=256,
        ae_n_enc_3=latent_dim,
        ae_n_dec_1=latent_dim,
        ae_n_dec_2=256,
        ae_n_dec_3=128,
        gae_n_enc_1=128,
        gae_n_enc_2=256,
        gae_n_enc_3=latent_dim,
        gae_n_dec_1=latent_dim,
        gae_n_dec_2=256,
        gae_n_dec_3=128,
        n_input=args.n_input,
        n_input1=args.n_input1,
        sigma=args.sigma,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        v=args.freedom_degree,
        fusion_mode=getattr(args, "fusion_mode", "concat"),
        fusion_hidden_dim=getattr(args, "fusion_hidden_dim", None),
        graph_encoder_variant="gcnii",
        gcnii_alpha=getattr(args, "gcnii_alpha", 0.3),
        gcnii_lambda=getattr(args, "gcnii_lambda", 0.5),
        use_boundary_aware_gcn=0,
    )
