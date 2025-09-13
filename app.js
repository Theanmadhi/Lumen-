import dotenv from "dotenv";
dotenv.config();
import mongoose from "mongoose";
import express from "express"
import cors from "cors"
import userRoutes from "./routes/User.route.js"
import adminRoutes from "./routes/Admin.route.js";
import { dashboard } from "./routes/dashboard.route.js";
import { protectRoute } from "./middlewares/protectRoute.js";
const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({extended: true}));

const MONGO_URL = process.env.MONGO_URL;

app.use("/users",userRoutes);
app.get("/dashboard",dashboard);
app.use("/admin",adminRoutes);

const start = async ()=>{
    const connectionDb = await mongoose.connect(MONGO_URL)
    console.log(`mongo connected DB host: ${connectionDb.connection.host}`)
    app.listen("8080",()=>{
    console.log("App is listening to 8080");
})
}
start();