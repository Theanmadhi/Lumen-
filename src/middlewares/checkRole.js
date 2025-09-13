
const checkRole = (role)=>{
    if(req.user.role == role){
        next();
    }
    else{
        return res.json({message:"The role is wrong"});
    }
}